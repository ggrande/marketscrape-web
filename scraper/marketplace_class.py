from __future__ import annotations

import datetime
import json
import logging
import re
from typing import Any

import numpy as np
from bs4 import BeautifulSoup

from .exceptions import InvalidDataFormat

logger = logging.getLogger(__name__)


class FacebookMarketplaceScraper:
    def __init__(self, mobile_soup: BeautifulSoup, base_soup: BeautifulSoup):
        self.mobile_soup = mobile_soup
        self.base_soup = base_soup

        script_tag = self.base_soup.find_all("script", {"type": "application/ld+json"})
        json_content: dict[str, Any] = {}

        for script in script_tag:
            script_content = script.string or ""
            try:
                parsed_content = json.loads(script_content)
            except json.JSONDecodeError:
                continue

            if isinstance(parsed_content, dict):
                json_content.update(parsed_content)
            elif isinstance(parsed_content, list):
                for item in parsed_content:
                    if isinstance(item, dict):
                        json_content.update(item)

        self.json_content = json_content
        self._validate_schema()

    def _validate_schema(self) -> None:
        required_keys = ["offers", "name", "description", "itemListElement"]
        missing = [key for key in required_keys if key not in self.json_content]
        if missing:
            raise InvalidDataFormat(
                f"Listing metadata is missing the following keys: {', '.join(missing)}"
            )

        item_list = self.json_content["itemListElement"]
        if not isinstance(item_list, list) or len(item_list) < 3:
            raise InvalidDataFormat("Listing metadata is incomplete.")

        offers = self.json_content["offers"]
        if not isinstance(offers, dict) or "price" not in offers or "priceCurrency" not in offers:
            raise InvalidDataFormat("Offer metadata is incomplete.")

    def get_listing_price(self) -> float:
        try:
            return float(self.json_content["offers"]["price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidDataFormat("Unable to determine listing price.") from exc

    def get_listing_title(self) -> str:
        try:
            return str(self.json_content["name"])
        except KeyError as exc:
            raise InvalidDataFormat("Listing title is missing.") from exc

    def get_listing_description(self) -> str:
        try:
            return str(self.json_content["description"])
        except KeyError as exc:
            raise InvalidDataFormat("Listing description is missing.") from exc

    def get_listing_city(self) -> str:
        try:
            return str(self.json_content["itemListElement"][1]["name"])
        except (KeyError, IndexError, TypeError) as exc:
            raise InvalidDataFormat("Listing city is unavailable.") from exc

    def get_listing_condition(self) -> str:
        item_conditions = {
            "New": 43.21,
            "Used - Like New": 29.15,
            "Used - Good": 25.33,
            "Used - Fair": 2.16,
            "Refurbished": 0.15,
        }

        schema = self.json_content.get("itemCondition")
        if schema and schema.replace("https://schema.org/", "") == "NewCondition":
            return "New"

        probabilities = [value / 100 for value in item_conditions.values()]
        return str(np.random.choice(list(item_conditions.keys()), p=probabilities))

    def get_listing_category(self) -> str:
        try:
            return str(self.json_content["itemListElement"][2]["name"])
        except (KeyError, IndexError, TypeError) as exc:
            raise InvalidDataFormat("Listing category is unavailable.") from exc

    def get_listing_image(self) -> str:
        images = self.mobile_soup.find_all("img")
        for image in images:
            src = image.get("src")
            if src and "https://scontent" in src:
                return src
        raise InvalidDataFormat("Listing image could not be located.")

    def get_listing_currency(self) -> str:
        try:
            return str(self.json_content["offers"]["priceCurrency"])
        except (KeyError, TypeError) as exc:
            raise InvalidDataFormat("Listing currency is missing.") from exc

    def get_listing_date(self) -> tuple[int, int]:
        tag = self.mobile_soup.find("abbr")
        if not tag or not tag.text:
            raise InvalidDataFormat("Listing date information is unavailable.")

        tag_text = tag.text.strip()

        try:
            month_str = re.search(r"[a-zA-Z]+", tag_text).group(0)
            month_num = datetime.datetime.strptime(month_str, "%B").month
        except (AttributeError, ValueError) as exc:
            hour_match = re.search(r"[0-9]+", tag_text)
            if hour_match:
                return 0, int(hour_match.group(0))
            raise InvalidDataFormat("Unable to parse listing date.") from exc

        year_match = re.search(r"[0-9]{4}", tag_text)
        year_str = year_match.group(0) if year_match else datetime.datetime.now().year

        date_match = re.search(r"[0-9]+", tag_text)
        time_match = re.search(r"[0-9]+:[0-9]+", tag_text)
        am_pm_match = re.search(r"[A-Z]{2}", tag_text)
        if not (date_match and time_match and am_pm_match):
            raise InvalidDataFormat("Listing date time components are incomplete.")

        formatted_time = f"{time_match.group(0)}:00 {am_pm_match.group(0)}"
        formatted_date = f"{year_str}-{month_num}-{date_match.group(0)}"
        dt_str = f"{formatted_date} {formatted_time}"
        formatted_dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %I:%M:%S %p")

        now = datetime.datetime.now()
        diff = now - formatted_dt

        days = diff.days
        hours = diff.seconds // 3600

        return days, hours

    def is_listing_missing(self) -> bool:
        title_element = self.mobile_soup.find("title")
        title_text = title_element.get_text().strip().lower() if title_element else ""

        text_to_find = "Buy and sell things locally on Facebook Marketplace."
        found = self.mobile_soup.find(string=text_to_find)

        return title_text == "page not found" or found is not None
