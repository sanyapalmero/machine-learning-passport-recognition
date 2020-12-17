import requests
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options

# Url to the yandex search with first opened image
TRAIN_PASSPORTS_URL = "https://yandex.ru/images/search?text=%D0%BF%D0%B0%D1%81%D0%BF%D0%BE%D1%80%D1%82&rpt=imagelike&url=https%3A%2F%2Favatars.mds.yandex.net%2Fget-images-cbir%2F4281001%2FIGE2JP0OCU-9L7QyVMJC3g3422%2Forig&cbir_id=4281001%2FIGE2JP0OCU-9L7QyVMJC3g3422&pos=0&img_url=https%3A%2F%2Fsun9-76.userapi.com%2Fc855616%2Fv855616049%2Fe6c4e%2FURwWxdSD0-Y.jpg"
TEST_PASSPORTS_URL = "https://yandex.ru/images/search?text=%D0%BF%D0%B0%D1%81%D0%BF%D0%BE%D1%80%D1%82&rpt=imagelike&url=https%3A%2F%2Fgosgo.ru%2Fwp-content%2Fuploads%2F2019%2F04%2Fpasport-rf-v-razvorote-737x1024.jpg&cbir_id=4258278%2Fbe0L4cs6Sl5QqnBZD9Ts0w2264&pos=0&img_url=https%3A%2F%2Fxn---44-ndd4bllir5g.xn--p1ai%2Fimg%2F053b1039b9d29f4e4bd73f980f0de4ec.jpg"
RANDOM_IMAGES_URL = "https://yandex.ru/images/search?text=%D1%81%D0%BB%D1%83%D1%87%D0%B0%D0%B9%D0%BD%D1%8B%D0%B5%20%D0%BA%D0%B0%D1%80%D1%82%D0%B8%D0%BD%D0%BA%D0%B8%20%D0%BD%D0%B0%20%D1%80%D0%B0%D0%B1%D0%BE%D1%87%D0%B8%D0%B9%20%D1%81%D1%82%D0%BE%D0%BB&pos=0&img_url=https%3A%2F%2Fsun9-14.userapi.com%2Fc638221%2Fv638221440%2F3d163%2FIPjN59HRNfI.jpg&rpt=simage"

# Count images to download
PASSPORTS_COUNT_TO_DOWNLOAD = 100

# If true - new chrome window will opened and all process will be displayed
DISPLAY_PROCESS = False

# Folder for saving downloaded passports
TRAIN_PASSPORTS_FOLDER = "passports-training-dataset/"
TEST_PASSPORTS_FOLDER = "passports-test-dataset/"


def download_and_save_passport_to_folder(file_name, file_url):
    response = requests.get(url=file_url)
    with open(TEST_PASSPORTS_FOLDER + file_name, "wb+") as file:
        file.write(response.content)
        print(f"File {file_name} was successfully saved")


def switch_to_next_image(driver):
    next_image_icon = driver.find_element_by_css_selector("div.CircleButton_type_next")
    next_image_icon.click()
    print("Switched to another image")


def run_script():
    chrome_options = Options()
    if not DISPLAY_PROCESS:
        chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(RANDOM_IMAGES_URL)

    for i in range(PASSPORTS_COUNT_TO_DOWNLOAD):
        try:
            img = driver.find_element_by_css_selector("img.MMImage-Origin")
        except NoSuchElementException:
            print(f"NoSuchElementException occured on {i} iteration.")
            break

        src = img.get_attribute("src")
        file_name = f"{i}_not_passport.jpg"

        download_and_save_passport_to_folder(file_name, src)
        switch_to_next_image(driver)


if __name__ == "__main__":
    run_script()
