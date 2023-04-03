from selenium import webdriver
import io
import os
import requests
import time
import hashlib
from PIL import Image


def fetch_image_urls(query: str, max_links_to_fetch: int, wd: webdriver, sleep_between_interactions: int = 1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeifind_elements_by_css_selectorght);")
        time.sleep(sleep_between_interactions)

        # build the Google query

    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements("css selector", "img.Q4LuWd")
        number_results = len(thumbnail_results)

        if number_results == results_start:
            return image_urls

        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls    
            actual_images = wd.find_elements("css selector", 'img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(30)
            load_more_button = wd.find_element("css selector", ".mye4qd")
            print(load_more_button)
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls


def persist_image(folder_path: str, url: str):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(folder_path, hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")


def search_and_download(search_term: str, driver_path: str, target_path: str, number_images: int):
    target_folder = os.path.join(target_path, '_'.join(search_term.lower().split(' ')))

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with webdriver.Chrome(executable_path=driver_path) as wd:
        res = fetch_image_urls(search_term, max_links_to_fetch=number_images, wd=wd, sleep_between_interactions=0.5)

    if (res):
        for elem in res:
            persist_image(target_folder, elem)


# Initialize the variables
DRIVER: str = input("Enter the absolute path of the chrome-driver: ")
TARGET: str = input("Enter the target path of the folder: ")
number_of_images = int(input("Enter number of images to be scrapped per query: "))
search_terms = []

print("\n")

# Query List
while True:
    query = input("Enter the seach query to be scarpped | Enter 'exit' to start scrapping the query list : ")
    if query == "exit":
        break;
    else:
        search_terms.append(query)

# Scarp The Images
for query in search_terms:
    search_and_download(query, DRIVER, TARGET, number_images=number_of_images)
