import requests
import pandas as pd
from bs4 import BeautifulSoup

def scrape_yelp_main_page(url):
    """
    Scrapes the main page of a Yelp restaurant and extracts:
    - Restaurant Name
    - Address
    - Phone Number
    - Website
    - Star Rating
    - Total Reviews
    - Cuisine Category
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch the page. Status code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract Restaurant Name
    try:
        restaurant_name = soup.find("h1").text.strip()
    except AttributeError:
        restaurant_name = "Unknown"

    # Extract Address
    try:
        address = soup.find("address").text.strip()
    except AttributeError:
        address = "Not Available"

    # Extract Phone Number
    try:
        phone = soup.find("p", class_="css-1p9ibgf").text.strip()
    except AttributeError:
        phone = "Not Available"

    # Extract Website (if available)
    try:
        website_tag = soup.find("a", class_="css-1um3nx")
        website = website_tag["href"] if website_tag else "Not Available"
    except AttributeError:
        website = "Not Available"

    # Extract Star Rating
    try:
        rating_tag = soup.find("div", class_="i-stars")
        rating = rating_tag["aria-label"] if rating_tag else "Not Available"
    except AttributeError:
        rating = "Not Available"

    # Extract Total Reviews
    try:
        reviews_tag = soup.find("span", class_="css-chan6m")
        reviews_count = reviews_tag.text.strip() if reviews_tag else "0"
    except AttributeError:
        reviews_count = "0"

    # Extract Cuisine Category
    try:
        category_tags = soup.find_all("span", class_="css-ardur")
        category = ", ".join([cat.text for cat in category_tags])
    except AttributeError:
        category = "Not Available"

    # Save data in dictionary
    restaurant_data = {
        "Restaurant Name": restaurant_name,
        "Address": address,
        "Phone Number": phone,
        "Website": website,
        "Star Rating": rating,
        "Total Reviews": reviews_count,
        "Cuisine Category": category
    }

    return restaurant_data

def save_to_csv(data, filename="C:/Users/Owner/Downloads/yelp_main_page.csv"):
    """
    Saves scraped data to a CSV file.
    """
    df = pd.DataFrame([data])
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"✅ Data successfully saved to {filename}")

if __name__ == "__main__":
    # Take URL as input
    restaurant_url = input("Enter Yelp restaurant URL: ").strip()

    # Scrape Yelp page
    restaurant_data = scrape_yelp_main_page(restaurant_url)

    if restaurant_data:
        save_to_csv(restaurant_data)
    else:
        print("⚠ Failed to scrape data.")