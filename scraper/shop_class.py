from __future__ import annotations

import logging
import re
import string
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Tuple

from . import utils
from .exceptions import InvalidDataFormat, InvalidSimilarityThreshold, NoProductsFound

logger = logging.getLogger(__name__)


class EbayScraper:
    def __init__(self) -> None:
        self.title: str | None = None
        self.start: int | None = None
        self.soup = None

    def create_url(self) -> None:
        """Populate ``self.soup`` with the results from an eBay search."""

        if self.title is None or self.start is None:
            raise InvalidDataFormat("Both title and start index must be initialised before scraping.")

        url = (
            "https://www.ebay.com/sch/i.html?_from=R40&_nkw="
            f"{self.title}&_sacat=0&_ipg=240&_pgn={self.start}"
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582"
            ),
            "Referer": "https://www.google.com/",
        }
        logger.debug("Fetching eBay search page %s", url)
        self.soup = utils.create_soup(url, headers)

    def _require_soup(self) -> None:
        if self.soup is None:
            raise InvalidDataFormat("No HTML document is loaded for parsing.")

    def get_product_title(self) -> List[str]:
        self._require_soup()
        return [element.text for element in self.soup.find_all("div", class_="s-item__title")]

    def get_product_price(self) -> List[float]:
        self._require_soup()
        prices: List[float] = []
        for element in self.soup.find_all("span", class_="s-item__price"):
            match = re.search(r"([0-9]+\.[0-9]+)|([0-9]+,[0-9]+)", element.text)
            if match:
                value = match.group(0).replace(",", "")
                prices.append(float(value))
        return prices

    def get_product_condition(self) -> List[str]:
        self._require_soup()
        return [element.text for element in self.soup.find_all("span", class_="SECONDARY_INFO")]

    def get_product_shipping(self) -> List[float]:
        self._require_soup()
        shipping_costs: List[float] = []
        for element in self.soup.find_all("span", class_="s-item__shipping s-item__logisticsCost"):
            match = re.search(r"([0-9]+.*[0-9])|(Free)|(not specified)", element.text)
            if not match:
                shipping_costs.append(0.0)
                continue
            if match.group(1):
                price = match.group(1).replace(",", "")
                shipping_costs.append(float(price))
            else:
                shipping_costs.append(0.0)
        return shipping_costs

    def get_product_country(self) -> List[str]:
        self._require_soup()
        countries: List[str] = []
        for element in self.soup.find_all("span", class_="s-item__location s-item__itemLocation"):
            countries.append(element.text.replace("from ", ""))
        return countries

    @staticmethod
    def get_similarity(string1: str, string2: str) -> float:
        string1 = string1.lower().strip()
        string2 = string2.lower().strip()

        translator = str.maketrans("", "", string.punctuation)
        string1_clean = string1.translate(translator)
        string2_clean = string2.translate(translator)

        similarity = SequenceMatcher(None, string1_clean, string2_clean).ratio()
        return similarity

    @staticmethod
    def remove_outliers(
        titles: List[str],
        prices: List[float],
        shipping: List[float],
        countries: List[str],
        conditions: List[str],
    ) -> Tuple[List[str], List[float], List[float], List[str], List[str]]:
        removal_threshold = 100

        if len(titles) >= removal_threshold:
            outlier_indices = utils.reject_outliers(prices, m=1.5)
            titles = [title for i, title in enumerate(titles) if i not in outlier_indices]
            prices = [price for i, price in enumerate(prices) if i not in outlier_indices]
            shipping = [ship for i, ship in enumerate(shipping) if i not in outlier_indices]
            countries = [country for i, country in enumerate(countries) if i not in outlier_indices]
            conditions = [condition for i, condition in enumerate(conditions) if i not in outlier_indices]

        return titles, prices, shipping, countries, conditions

    def get_product_info(self) -> List[dict]:
        titles = self.get_product_title()
        prices = self.get_product_price()
        shipping = self.get_product_shipping()
        countries = self.get_product_country()
        conditions = self.get_product_condition()

        titles, prices, shipping, countries, conditions = self.remove_outliers(
            titles, prices, shipping, countries, conditions
        )

        product_info = []
        for title, price, ship, country, condition in zip(titles, prices, shipping, countries, conditions):
            product_info.append(
                {
                    "title": utils.clean_text(title.lower()),
                    "price": price,
                    "shipping": ship,
                    "country": country,
                    "condition": condition,
                }
            )

        return product_info

    @staticmethod
    def lowest_price_highest_similarity(filtered_prices_descriptions: Dict[str, dict]) -> Tuple[str, dict]:
        if not filtered_prices_descriptions:
            raise NoProductsFound("No comparable products were found.")

        max_similarity_item = None
        min_price_item = None

        for _, item_details in filtered_prices_descriptions.items():
            if max_similarity_item is None and min_price_item is None:
                max_similarity_item = item_details
                min_price_item = item_details
            else:
                if item_details["similarity"] > max_similarity_item["similarity"]:
                    max_similarity_item = item_details
                if item_details["price"] < min_price_item["price"]:
                    min_price_item = item_details

        max_similar_items = [
            (item_name, item_details)
            for item_name, item_details in filtered_prices_descriptions.items()
            if item_details["similarity"] == max_similarity_item["similarity"]
        ]

        min_price_item = min(max_similar_items, key=lambda x: x[1]["price"])

        return min_price_item

    @staticmethod
    def construct_candidates(
        descriptions: Iterable[str],
        prices: Iterable[str],
        shipping: Iterable[str],
        countries: Iterable[str],
        conditions: Iterable[str],
        similarities: Iterable[float],
    ) -> Dict[str, dict]:
        description_list = list(descriptions)
        price_list = list(prices)
        shipping_list = list(shipping)
        country_list = list(countries)
        condition_list = list(conditions)
        similarity_list = list(similarities)

        lengths = {
            len(description_list),
            len(price_list),
            len(shipping_list),
            len(country_list),
            len(condition_list),
            len(similarity_list),
        }
        if len(lengths) != 1:
            raise InvalidDataFormat("Candidate lists must all be the same length.")

        candidates: Dict[str, dict] = {}
        for description, price, ship, country, condition, similarity in zip(
            description_list, price_list, shipping_list, country_list, condition_list, similarity_list
        ):
            candidates[description] = {
                "price": price,
                "shipping": ship,
                "country": country,
                "condition": condition,
                "similarity": similarity,
            }

        return candidates

    def find_viable_product(self, title: str, ramp_down: float) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[float]]:
        descriptions: List[str] = []
        prices: List[str] = []
        shipping: List[str] = []
        countries: List[str] = []
        conditions: List[str] = []
        similarities: List[float] = []

        for page_number in range(5):
            similarity_threshold = 0.35
            self.title = title
            self.start = page_number
            self.create_url()

            try:
                filtered_prices_descriptions = self.listing_product_similarity(title, similarity_threshold)
                if not filtered_prices_descriptions:
                    raise NoProductsFound("No similar products found")
            except NoProductsFound:
                consecutively_empty = 0
                filtered_prices_descriptions = {}
                while not filtered_prices_descriptions:
                    ramp_down += 0.05
                    filtered_prices_descriptions = self.listing_product_similarity(
                        title, max(0.0, similarity_threshold - ramp_down)
                    )
                    if consecutively_empty == 2:
                        break

                    if filtered_prices_descriptions:
                        consecutively_empty = 0
                    else:
                        consecutively_empty += 1

            descriptions += list(filtered_prices_descriptions.keys())
            prices += [f"{product['price']:,.2f}" for product in filtered_prices_descriptions.values()]
            shipping += [f"{product['shipping']:,.2f}" for product in filtered_prices_descriptions.values()]
            countries += [product["country"] for product in filtered_prices_descriptions.values()]
            conditions += [product["condition"] for product in filtered_prices_descriptions.values()]
            similarities += [product["similarity"] for product in filtered_prices_descriptions.values()]

        if not descriptions:
            raise NoProductsFound("No viable products were discovered.")

        return descriptions, prices, shipping, countries, conditions, similarities

    def filter_products_by_similarity(
        self,
        product_info: List[dict],
        target_title: str,
        similarity_threshold: float,
    ) -> Dict[str, dict]:
        if not 0 <= similarity_threshold <= 1:
            raise InvalidSimilarityThreshold("Similarity threshold must be between 0 and 1.")

        filtered_products: Dict[str, dict] = {}
        for product in product_info:
            similarity = self.get_similarity(product["title"], target_title)
            if similarity >= similarity_threshold:
                filtered_products[product["title"]] = {
                    "price": product["price"],
                    "shipping": product["shipping"],
                    "country": product["country"],
                    "condition": product["condition"],
                    "similarity": similarity,
                }

        return filtered_products

    def listing_product_similarity(self, title: str, similarity_threshold: float) -> Dict[str, dict]:
        product_info = self.get_product_info()
        return self.filter_products_by_similarity(product_info, title.lower(), similarity_threshold)
