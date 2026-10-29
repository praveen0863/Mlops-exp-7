import unittest
from unittest.mock import patch

from app import app


class TestLogging(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    @patch("app.logger.info")
    def test_health_check_logging(self, mock_logger):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        mock_logger.assert_called()

    @patch("app.logger.warning")
    def test_invalid_request_logging(self, mock_logger):
        response = self.client.post(
            "/predict",
            json={
                "attendance": 90
            }
        )

        self.assertEqual(response.status_code, 400)
        mock_logger.assert_called()


if __name__ == "__main__":
    unittest.main()
