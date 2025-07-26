import requests
from bs4 import BeautifulSoup
import csv
import re
import math


def scrape_page(url):
    """
    Scrapes a single page of reviews from a Yelp restaurant page.
    Returns a tuple: (restaurant_name, total_reviews_text, reviews)
      - restaurant_name: string (if found)
      - total_reviews_text: the raw text containing the review count (e.g. "3788 reviews")
      - reviews: a list of reviews where each review is a list:
            [review_text, reviewer, rating]
    """
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve webpage. Status code: {response.status_code}")
            return None, None, None

        page_content = response.content
        soup = BeautifulSoup(page_content, 'html.parser')

        # Extract restaurant name (adjust the tag and class as needed)
        restaurant_name_tag = soup.find('h1', {'class': 'y-css-olzveb'})
        restaurant_name = restaurant_name_tag.get_text(strip=True) if restaurant_name_tag else 'Unknown Restaurant'

        # Extract total reviews text (e.g., "3788 reviews")
        total_reviews_tag = soup.find('span', {'class': 'y-css-yrt0i5'})
        total_reviews_text = total_reviews_tag.get_text(strip=True) if total_reviews_tag else '0'

        # Extract all review elements (each review is in an <li> with class "y-css-1sqelp2")
        review_elements = soup.find_all('li', {'class': 'y-css-1sqelp2'})
        if not review_elements:
            print("No reviews found on this page.")
            return restaurant_name, total_reviews_text, None

        reviews = []
        for review in review_elements:
            # Extract review text
            review_text_tag = review.find('span', {'class': 'raw__09f24__T4Ezm'})
            review_text = review_text_tag.get_text(strip=True) if review_text_tag else 'No review text'

            # Extract reviewer name
            reviewer_tag = review.find('a', {'class': 'y-css-1x1e1r2'})
            reviewer = reviewer_tag.get_text(strip=True) if reviewer_tag else 'Unknown reviewer'

            # Determine the rating by counting the <div> elements with class "y-css-16lknu1"
            rating_divs = review.find_all('div', {'class': 'y-css-16lknu1'})
            rating = len(rating_divs)

            reviews.append([review_text, reviewer, rating])

        return restaurant_name, total_reviews_text, reviews

    except Exception as e:
        print(f"An error occurred while scraping the page: {e}")
        return None, None, None


def main():
    base_url = input("Enter the base URL of the restaurant review page: ").strip()


    # First, scrape the first page to get restaurant info and total reviews
    restaurant_name, total_reviews_text, first_page_reviews = scrape_page(base_url)
    if total_reviews_text is None:
        print("Could not determine the total review count.")
        return

    # Parse the total number of reviews from the total_reviews_text (e.g., "3788 reviews")
    m = re.search(r'(\d+)', total_reviews_text)
    if m:
        total_reviews_count = int(m.group(1))
    else:
        total_reviews_count = 0

    print(f"Restaurant: {restaurant_name} with {total_reviews_count} reviews.")

    # Assume Yelp shows 10 reviews per page
    reviews_per_page = 10
    max_pages = math.ceil(total_reviews_count / reviews_per_page)
    print(f"Calculated total pages to scrape: {max_pages}")

    # Prepare CSV filename based on the restaurant name
    filename = f"{restaurant_name.replace(' ', '_')}_reviews.csv"
    # Write header to CSV
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Restaurant Name', 'Total Reviews', 'Review Text', 'Reviewer', 'Rating'])

    # Save reviews from the first page (if any)
    if first_page_reviews:
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for review in first_page_reviews:
                writer.writerow([restaurant_name, total_reviews_count] + review)

    # Loop over remaining pages (page 2 to max_pages)
    for page in range(1, max_pages):
        page_number = page + 1
        # Construct the URL: append "&start={page*10}" to the base URL
        url = f"{base_url}&start={page * reviews_per_page}"
        print(f"Scraping page {page_number}: {url}")
        _, _, reviews = scrape_page(url)
        if not reviews:
            print(f"No reviews found on page {page_number}. Ending pagination.")
            break

        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for review in reviews:
                writer.writerow([restaurant_name, total_reviews_count] + review)
        # Optional: add a small delay between requests
        # time.sleep(1)

    print(f"All reviews have been saved to {filename}.")


if __name__ == '__main__':
    main()
