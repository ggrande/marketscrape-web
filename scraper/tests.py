import datetime
import json
from unittest import mock

import requests
from bs4 import BeautifulSoup
from django.test import SimpleTestCase

from . import utils
from .exceptions import InvalidDataFormat, InvalidSimilarityThreshold, ScraperRequestError
from .marketplace_class import FacebookMarketplaceScraper
from .shop_class import EbayScraper


class UtilsTests(SimpleTestCase):
    def test_remove_illegal_characters_replaces_reserved(self):
        result = utils.remove_illegal_characters("Guitar #1 & Case")
        self.assertEqual(result, "Guitar %21 %26 Case")

    def test_clean_text_normalises_whitespace(self):
        result = utils.clean_text("   Hello!!!   World\n")
        self.assertEqual(result, "Hello World")

    @mock.patch("scraper.utils.requests.get")
    def test_create_soup_success(self, mock_get):
        mock_response = mock.Mock()
        mock_response.text = "<html><body><p>ok</p></body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        soup = utils.create_soup("https://example.com")
        self.assertEqual(soup.find("p").text, "ok")

    @mock.patch("scraper.utils.requests.get", side_effect=requests.RequestException("boom"))
    def test_create_soup_failure(self, _mock_get):
        with self.assertRaises(ScraperRequestError):
            utils.create_soup("https://example.com")

    def test_reject_outliers_returns_indices(self):
        data = [1, 1, 1, 10]
        indices = utils.reject_outliers(data, m=1.5)
        self.assertEqual(indices, [3])

    def test_price_difference_rating_penalises_old_listings(self):
        rating = utils.price_difference_rating(100.0, 50.0, days=14)
        self.assertLess(rating, 5.0)

    def test_percentage_difference_handles_equal(self):
        result = utils.percentage_difference(100.0, 100.0)
        self.assertEqual(result, {"amount": "0.00", "type": "equal"})

    def test_create_chart_produces_json(self):
        json_chart = utils.create_chart(
            similar_prices=[100.0, 110.0, 120.0],
            similar_shipping=[10.0, 12.0, 8.0],
            similar_descriptions=["alpha", "beta", "gamma"],
            similar_conditions=["New", "Used", "New"],
            listing_currency="USD",
            listing_title="Sample",
            best_title="beta",
        )
        payload = json.loads(json_chart)
        self.assertIn("data", payload)
        self.assertIn("layout", payload)

    def test_create_bargraph_with_empty_data(self):
        json_bar = utils.create_bargraph([])
        payload = json.loads(json_bar)
        self.assertIn("data", payload)
        self.assertEqual(payload["data"], [])


class EbayScraperTests(SimpleTestCase):
    def test_construct_candidates_requires_equal_lengths(self):
        with self.assertRaises(InvalidDataFormat):
            EbayScraper.construct_candidates(["a"], [], [], [], [], [])

    def test_filter_products_by_similarity_threshold(self):
        scraper = EbayScraper()
        product_info = [
            {"title": "test item", "price": 10.0, "shipping": 2.0, "country": "US", "condition": "New"}
        ]
        result = scraper.filter_products_by_similarity(product_info, "test item", similarity_threshold=0.5)
        self.assertIn("test item", result)

    def test_filter_products_by_similarity_invalid_threshold(self):
        scraper = EbayScraper()
        with self.assertRaises(InvalidSimilarityThreshold):
            scraper.filter_products_by_similarity([], "title", similarity_threshold=1.5)


class MarketplaceScraperTests(SimpleTestCase):
    def setUp(self):
        self.base_html = """
        <html>
            <head>
                <script type=\"application/ld+json\">
                {
                    \"name\": \"Vintage Camera\",
                    \"description\": \"Great condition\",
                    \"offers\": {\"price\": 120.0, \"priceCurrency\": \"USD\"},
                    \"itemListElement\": [
                        {\"name\": \"Marketplace\"},
                        {\"name\": \"New York\"},
                        {\"name\": \"Electronics\"}
                    ],
                    \"itemCondition\": \"https://schema.org/NewCondition\"
                }
                </script>
            </head>
            <body></body>
        </html>
        """
        self.mobile_html = """
        <html>
            <head><title>Marketplace Listing</title></head>
            <body>
                <abbr>January 1, 2024 at 1:00 PM</abbr>
                <img src=\"https://scontent.example.com/img.jpg\" />
            </body>
        </html>
        """

    def test_parses_listing_metadata(self):
        base_soup = BeautifulSoup(self.base_html, "html.parser")
        mobile_soup = BeautifulSoup(self.mobile_html, "html.parser")
        scraper = FacebookMarketplaceScraper(mobile_soup, base_soup)

        self.assertEqual(scraper.get_listing_price(), 120.0)
        self.assertEqual(scraper.get_listing_title(), "Vintage Camera")
        self.assertEqual(scraper.get_listing_city(), "New York")
        self.assertEqual(scraper.get_listing_category(), "Electronics")
        self.assertEqual(scraper.get_listing_currency(), "USD")
        self.assertEqual(scraper.get_listing_image(), "https://scontent.example.com/img.jpg")
        self.assertEqual(scraper.get_listing_condition(), "New")

        real_datetime = datetime.datetime
        with mock.patch("scraper.marketplace_class.datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = real_datetime(2024, 1, 2, 13, 0)
            mock_datetime.strptime.side_effect = lambda *args, **kwargs: real_datetime.strptime(*args, **kwargs)
            days, hours = scraper.get_listing_date()
        self.assertEqual(days, 1)

    def test_missing_listing_detection(self):
        base_soup = BeautifulSoup(self.base_html, "html.parser")
        missing_mobile_html = """
        <html>
            <head><title>Page not found</title></head>
            <body>Buy and sell things locally on Facebook Marketplace.</body>
        </html>
        """
        mobile_soup = BeautifulSoup(missing_mobile_html, "html.parser")
        scraper = FacebookMarketplaceScraper(mobile_soup, base_soup)
        self.assertTrue(scraper.is_listing_missing())
