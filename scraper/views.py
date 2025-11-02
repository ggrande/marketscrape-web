from __future__ import annotations

import logging
import re

from django.shortcuts import render
from django.views import View

from . import utils
from .exceptions import InvalidDataFormat, NoProductsFound, ScraperRequestError
from .forms import MarketForm
from .marketplace_class import FacebookMarketplaceScraper
from .shop_class import EbayScraper

logger = logging.getLogger(__name__)


class Index(View):
    def get(self, request):
        form = MarketForm()
        context = {"form": form}
        return render(request, "scraper/index.html", context)

    def post(self, request):
        form = MarketForm(request.POST)
        if not form.is_valid():
            return render(request, "scraper/index.html", {"form": form})

        url = form.cleaned_data["url"]
        match = re.search(r".*[0-9]", url)
        if not match:
            form.add_error("url", "The provided URL does not contain a listing identifier.")
            return render(request, "scraper/index.html", {"form": form})

        shortened_url = match.group(0)
        mobile_url = re.sub(r"//www\.", "//m.", shortened_url)

        try:
            mobile_soup = utils.create_soup(mobile_url)
            base_soup = utils.create_soup(url)
        except ScraperRequestError:
            logger.warning("Failed to fetch Marketplace listing %s", url, exc_info=True)
            form.add_error(None, "We couldn't reach Facebook Marketplace. Please try again later.")
            return render(request, "scraper/index.html", {"form": form})
        except InvalidDataFormat as exc:
            form.add_error("url", str(exc))
            return render(request, "scraper/index.html", {"form": form})

        try:
            facebook_instance = FacebookMarketplaceScraper(mobile_soup, base_soup)
        except InvalidDataFormat as exc:
            logger.warning("Marketplace listing metadata invalid for %s", url, exc_info=True)
            form.add_error(None, str(exc))
            return render(request, "scraper/index.html", {"form": form})

        if facebook_instance.is_listing_missing():
            return render(request, "scraper/missing.html")

        image = facebook_instance.get_listing_image()
        days, hours = facebook_instance.get_listing_date()
        description = facebook_instance.get_listing_description()
        title = facebook_instance.get_listing_title()
        condition = facebook_instance.get_listing_condition()
        category = facebook_instance.get_listing_category()
        price = facebook_instance.get_listing_price()
        city = facebook_instance.get_listing_city()
        currency = facebook_instance.get_listing_currency()

        shopping_instance = EbayScraper()

        cleaned_title = utils.remove_illegal_characters(title)
        try:
            (
                similar_descriptions,
                similar_prices,
                similar_shipping,
                similar_countries,
                similar_conditions,
                similar_scores,
            ) = shopping_instance.find_viable_product(cleaned_title, ramp_down=0.0)
        except NoProductsFound:
            form.add_error(None, "We couldn't find comparable listings on eBay.")
            return render(request, "scraper/index.html", {"form": form})

        candidates = shopping_instance.construct_candidates(
            similar_descriptions,
            similar_prices,
            similar_shipping,
            similar_countries,
            similar_conditions,
            similar_scores,
        )

        try:
            similar_prices_float = [float(price.replace(",", "")) for price in similar_prices]
            similar_shipping_float = [float(ship.replace(",", "")) for ship in similar_shipping]
        except ValueError:
            form.add_error(None, "We couldn't parse eBay pricing data.")
            return render(request, "scraper/index.html", {"form": form})

        try:
            best_product = shopping_instance.lowest_price_highest_similarity(candidates)
        except NoProductsFound:
            form.add_error(None, "We couldn't determine the best comparable listing.")
            return render(request, "scraper/index.html", {"form": form})

        best_description, best_details = best_product
        best_price = float(best_details["price"])
        best_shipping = float(best_details["shipping"])
        best_country = best_details["country"]
        best_score = best_details["similarity"] * 100

        try:
            idx = similar_countries.index(best_country)
        except ValueError:
            idx = 0

        best_price_display = f"{similar_prices_float[idx]:,.2f}"
        best_shipping_display = f"{similar_shipping_float[idx]:,.2f}"

        best_total = best_price + best_shipping
        best_context = utils.percentage_difference(price, best_total)
        price_rating = utils.price_difference_rating(price, best_total, days)

        chart = utils.create_chart(
            similar_prices_float,
            similar_shipping_float,
            similar_descriptions,
            similar_conditions,
            currency,
            title,
            best_description,
        )
        bargraph = utils.create_bargraph(similar_countries)

        total_items = len(similar_descriptions)

        context = {
            "shortened_url": shortened_url,
            "mobile_url": mobile_url,
            "title": title,
            "price": f"{float(price):,.2f}",
            "chart": chart,
            "bargraph": bargraph,
            "price_rating": round(price_rating, 1),
            "days": days,
            "hours": hours,
            "image": image,
            "description": description,
            "condition": condition,
            "category": category,
            "city": city,
            "currency": currency,
            "total_items": total_items,
            "best_price": best_price_display,
            "best_shipping": best_shipping_display,
            "best_title": best_description.title(),
            "best_score": round(best_score, 2),
            "best_context": best_context,
        }

        return render(request, "scraper/result.html", context)
