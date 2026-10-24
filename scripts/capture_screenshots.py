#!/usr/bin/env python3
"""
ShopSense - Automated UI Screenshot Capturing Script
Uses Playwright with local Chrome to capture high-resolution screenshots of the live application.
"""

import time
import os
import subprocess
from playwright.sync_api import sync_playwright

CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SCREENSHOT_DIR = "/Users/gitanshgarg/Documents/shopsense/screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def capture_ui_screenshots():
    print("Launching Streamlit server on port 8503...")
    st_proc = subprocess.Popen(
        ["/Users/gitanshgarg/Documents/shopsense/.venv/bin/streamlit", "run", "app.py",
         "--server.headless", "true", "--server.port", "8503"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(4)

    try:
        with sync_playwright() as p:
            print(f"Launching Chrome from: {CHROME_PATH}")
            browser = p.chromium.launch(
                executable_path=CHROME_PATH,
                headless=True
            )
            context = browser.new_context(viewport={"width": 1440, "height": 960}, device_scale_factor=1.5)
            page = context.new_page()

            url = "http://localhost:8503"
            print(f"Navigating to {url}...")
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(3000)

            # 1. Executive Overview Screenshot
            print("Capturing 01_executive_overview.png...")
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "01_executive_overview.png"), full_page=False)

            # 2. Sales & Products Tab
            print("Capturing 02_sales_and_products.png...")
            page.locator("text=Sales & Products").first.click()
            page.wait_for_timeout(2000)
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "02_sales_and_products.png"), full_page=False)

            # 3. Demand Forecast Tab
            print("Capturing 03_demand_forecast.png...")
            page.locator("text=Demand Forecast").first.click()
            page.wait_for_timeout(2500)
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "03_demand_forecast.png"), full_page=False)

            # 4. Inventory Intelligence Tab
            print("Capturing 04_inventory_intelligence.png...")
            page.locator("text=Inventory Intelligence").first.click()
            page.wait_for_timeout(2000)
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "04_inventory_intelligence.png"), full_page=False)

            # 5. Profitability & What-If Tab
            print("Capturing 05_profitability_simulator.png...")
            page.locator("text=Profitability & What-If").first.click()
            page.wait_for_timeout(2000)
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "05_profitability_simulator.png"), full_page=False)

            # 6. Ask ShopSense Tab
            print("Capturing 06_ask_shopsense_ai.png...")
            page.locator("text=Ask ShopSense").first.click()
            page.wait_for_timeout(1500)
            # Fill the question input and press Enter
            inp = page.locator("input[placeholder*='Which products are at risk']").first
            if inp.count() > 0:
                inp.fill("Which products are at critical risk of running out?")
                inp.press("Enter")
                page.wait_for_timeout(3000)
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "06_ask_shopsense_ai.png"), full_page=False)

            # 7. Data Quality & Methodology Tab
            print("Capturing 07_data_quality_methodology.png...")
            page.locator("text=Data Quality & Methodology").first.click()
            page.wait_for_timeout(2000)
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "07_data_quality_methodology.png"), full_page=False)

            browser.close()
            print("Successfully captured all 7 genuine UI screenshots!")
    finally:
        st_proc.terminate()
        st_proc.wait()

if __name__ == "__main__":
    capture_ui_screenshots()
