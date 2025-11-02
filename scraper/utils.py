from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from .exceptions import InvalidDataFormat, ScraperRequestError

logger = logging.getLogger(__name__)


def remove_illegal_characters(title: str) -> str:
    """Replace characters that would break Marketplace URLs."""

    if not isinstance(title, str):
        raise InvalidDataFormat("Title must be provided as a string.")

    sanitized = re.sub(r"#", "%2", title)
    sanitized = re.sub(r"&", "%26", sanitized)

    return sanitized


def clean_text(title: str) -> str:
    """Remove non alphanumeric characters and redundant whitespace."""

    if not isinstance(title, str):
        raise InvalidDataFormat("Title must be provided as a string.")

    cleaned = re.sub(r"[^A-Za-z0-9\s]+", " ", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def create_soup(url: str, headers: dict | None = None) -> BeautifulSoup:
    """Create a :class:`BeautifulSoup` object from a remote URL."""

    if not isinstance(url, str) or not url:
        raise InvalidDataFormat("A non-empty URL must be supplied.")

    request_headers = headers or {}
    try:
        logger.debug("Requesting URL %s", url)
        response = requests.get(url, headers=request_headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - exercised via tests
        logger.error("Unable to fetch %s", url, exc_info=exc)
        raise ScraperRequestError(f"Unable to fetch '{url}'") from exc

    return BeautifulSoup(response.text, "html.parser")


def reject_outliers(data: Sequence[float], m: float) -> list[int]:
    """Return the indices of items that lie outside ``m`` interquartile ranges."""

    data_array = np.asarray(data, dtype=float)
    if data_array.size == 0:
        logger.debug("reject_outliers received no data")
        return []

    distribution = np.abs(data_array - np.median(data_array))
    m_deviation = np.median(distribution)
    if np.isclose(m_deviation, 0):
        indices = np.where(distribution >= m)[0]
        return indices.tolist()

    standard = distribution / m_deviation
    indices = np.where(standard >= m)[0]

    return indices.tolist()


def price_difference_rating(initial: float, final: float, days: int) -> float:
    """Return a 0–5 rating that penalises listings that stay expensive for long."""

    try:
        initial_price = float(initial)
        final_price = float(final)
    except (TypeError, ValueError) as exc:
        raise InvalidDataFormat("Initial and final prices must be numeric.") from exc

    if initial_price <= 0:
        raise InvalidDataFormat("Initial price must be greater than zero.")

    decay_constant = 0.01
    linear_factor = 0.0125
    threshold_days = 7

    adjusted_initial = initial_price
    if days >= threshold_days:
        days_past_threshold = days - threshold_days
        penalty_amount = (
            initial_price * np.exp(-decay_constant * days_past_threshold)
            + linear_factor * days_past_threshold * initial_price
        )
        adjusted_initial += penalty_amount

    if adjusted_initial <= final_price:
        rating = 5.0
    else:
        price_difference = adjusted_initial - final_price
        rating = 5.0 - (price_difference / adjusted_initial) * 5.0

    return float(np.clip(rating, 0.0, 5.0))


def percentage_difference(list_price: float, best_price: float) -> dict[str, str]:
    """Return the relative difference between a listing price and the best price."""

    try:
        list_price_value = float(list_price)
        best_price_value = float(best_price)
    except (TypeError, ValueError) as exc:
        raise InvalidDataFormat("Prices must be numeric values.") from exc

    if list_price_value < 0 or best_price_value < 0:
        raise InvalidDataFormat("Prices must be zero or greater.")

    if np.isclose(list_price_value, best_price_value):
        return {"amount": "0.00", "type": "equal"}

    if list_price_value > best_price_value:
        baseline = list_price_value if not np.isclose(list_price_value, 0.0) else 1.0
        difference_type = "decrease"
        difference_value = list_price_value - best_price_value
    else:
        baseline = best_price_value if not np.isclose(best_price_value, 0.0) else 1.0
        difference_type = "increase"
        difference_value = best_price_value - list_price_value

    percentage = (np.abs(difference_value) / baseline) * 100

    return {"amount": f"{percentage:.2f}", "type": difference_type}


def _validate_chart_inputs(
    similar_prices: Sequence[float],
    similar_shipping: Sequence[float],
    similar_descriptions: Sequence[str],
    similar_conditions: Sequence[str],
) -> None:
    lengths = {
        "prices": len(similar_prices),
        "shipping": len(similar_shipping),
        "descriptions": len(similar_descriptions),
        "conditions": len(similar_conditions),
    }

    if len(set(lengths.values())) != 1:
        raise InvalidDataFormat(
            "All comparison collections must contain the same number of items."
        )

    if not lengths["prices"]:
        raise InvalidDataFormat("At least one comparison product is required.")


def create_chart(
    similar_prices: Sequence[float],
    similar_shipping: Sequence[float],
    similar_descriptions: Sequence[str],
    similar_conditions: Sequence[str],
    listing_currency: str,
    listing_title: str,
    best_title: str,
) -> str:
    """Return a JSON encoded Plotly chart visualising comparable listings."""

    _validate_chart_inputs(
        similar_prices, similar_shipping, similar_descriptions, similar_conditions
    )

    prices_array = np.asarray(similar_prices, dtype=float)
    shipping_array = np.asarray(similar_shipping, dtype=float)
    descriptions_array = np.asarray(similar_descriptions, dtype=str)
    conditions_array = np.asarray(similar_conditions, dtype=str)

    sorted_indices = np.argsort(shipping_array)
    sorted_prices = prices_array[sorted_indices].reshape(-1, 1)
    sorted_shipping = shipping_array[sorted_indices]
    sorted_descriptions = descriptions_array[sorted_indices]
    sorted_conditions = conditions_array[sorted_indices]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sorted_prices[:, 0],
            y=sorted_shipping,
            mode="markers",
            marker=dict(
                color=sorted_prices[:, 0] + sorted_shipping,
                colorscale="RdYlGn_r",
                colorbar=dict(title="Total Price"),
                size=8,
            ),
            hovertemplate="%{text}",
            text=[
                (
                    f"Product: {desc.title()}<br>Price: ${price:,.2f}<br>"
                    f"Shipping: ${ship:,.2f}<br>Condition: {cond}"
                )
                for desc, price, ship, cond in zip(
                    sorted_descriptions,
                    sorted_prices[:, 0],
                    sorted_shipping,
                    sorted_conditions,
                )
            ],
            showlegend=False,
            name="Products",
        )
    )

    try:
        best_index = int(np.where(sorted_descriptions == best_title)[0][0])
    except IndexError:
        logger.debug("Best title '%s' not found in descriptions; defaulting to first.", best_title)
        best_index = 0

    best_price = float(sorted_prices[best_index, 0])
    best_shipping = float(sorted_shipping[best_index])
    fig.add_trace(
        go.Scatter(
            x=[best_price],
            y=[best_shipping],
            mode="markers",
            marker=dict(color="#fc0", symbol="star", size=12),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    if sorted_prices.shape[0] > 1:
        poly_features = PolynomialFeatures(degree=4, include_bias=True)
        X_poly = poly_features.fit_transform(sorted_prices)
        poly_model = LinearRegression()
        poly_model.fit(X_poly, sorted_shipping)

        X_range = np.linspace(sorted_prices.min(), sorted_prices.max(), 100)
        X_range_poly = poly_features.fit_transform(X_range.reshape(-1, 1))
        Y_range = poly_model.predict(X_range_poly)

        y_pred = poly_model.predict(X_poly)
        mse = mean_squared_error(sorted_shipping, y_pred)
        ci = 1.96 * np.sqrt(mse)

        fig.add_trace(
            go.Scatter(
                x=X_range,
                y=Y_range,
                mode="lines",
                hovertemplate="%{text}",
                text=[
                    f"Predicted Price: ${price:.2f}<br>Predicted Shipping: ${ship:.2f}"
                    for price, ship in zip(X_range, Y_range)
                ],
                showlegend=False,
                name="Trend Line",
                line_color="rgb(128, 128, 128)",
                line=dict(dash="longdash"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([X_range, X_range[::-1]]),
                y=np.concatenate([Y_range + ci, Y_range[::-1] - ci]),
                fill="toself",
                fillcolor="rgba(128, 128, 128, 0.15)",
                line_color="rgba(255, 255, 255, 0)",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        template="plotly_white",
        hovermode="closest",
        xaxis_title=f"Product Price $({listing_currency})",
        yaxis_title=f"Shipping Cost $({listing_currency})",
        legend_title="Categories",
        title={
            "text": f"Products Similar to: {listing_title}",
            "xanchor": "center",
            "yanchor": "top",
            "y": 0.9,
            "x": 0.5,
        },
    )

    return fig.to_json()


def create_bargraph(countries: Sequence[str]) -> str:
    """Return a JSON encoded bar graph for the provided countries."""

    fig = go.Figure()
    if countries:
        country_counts = Counter(countries)
        country_names = list(country_counts.keys())
        country_values = list(country_counts.values())

        fig.add_trace(
            go.Bar(
                x=country_names,
                y=country_values,
                hoverinfo="text",
                hovertext=[
                    f"Country: {country}<br>Citations: {count}"
                    for country, count in zip(country_names, country_values)
                ],
                marker=dict(
                    color=country_values,
                    colorscale="RdYlGn_r",
                    showscale=True,
                    colorbar=dict(title="Citations"),
                ),
            )
        )
    else:
        logger.debug("create_bargraph received no country data")

    fig.update_layout(
        xaxis_title="Country of Origin",
        yaxis_title="Citations",
        title={
            "text": "Frequently Cited Countries",
            "xanchor": "center",
            "yanchor": "top",
            "y": 0.9,
            "x": 0.5,
        },
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig.to_json()
