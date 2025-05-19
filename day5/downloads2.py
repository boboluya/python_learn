from icrawler.builtin import GoogleImageCrawler

def download_negative_samples(keyword, max_num=10, output_dir='negative_samples'):
    google_crawler = GoogleImageCrawler(storage={'root_dir': f'{output_dir}/{keyword.replace(" ", "_")}'})
    google_crawler.crawl(keyword=keyword, max_num=max_num)

if __name__ == "__main__":
    download_negative_samples('empty room', max_num=100)
