"""
Job Scraper

This script scrapes job listings from a job listing website (TimesJobs).
It extracts details such as job title, company name, required skills, experience, 
salary, location, and a job listing link.

Main functionalities:
1. Fetches job listings from multiple pages.
2. Extracts key job details.
3. Retrieves additional details (experience, salary, skills) from individual job pages.
4. Saves extracted job data to a CSV file.

Usage:
- Run this script to scrape job postings and store them in `scraped_jobs.csv`.
"""

import requests
from bs4 import BeautifulSoup
import re
from job_details import JobDetails
from utils import save_to_csv


class JobScraper:
    """Scrapes job postings from a job listing website."""

    def __init__(self, url):
        """
        Initialize the scraper with the given URL.

        Args:
            url (str): The URL to scrape job listings from.
        """
        self.url = url
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
        }

    def fetch_jobs(self):
        """
        Fetch job listings from the given URL.

        Returns:
            list: List of extracted job data dictionaries.
        """
        response = requests.get(self.url, headers=self.headers)
        
        if response.status_code != 200:
            print(f"Failed to fetch jobs. Status code: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, 'lxml')
        jobs = soup.find_all('li', class_='clearfix job-bx wht-shd-bx') or []
        
        return self.extract_jobs(jobs)

    def extract_jobs(self, jobs):
        """
        Extract job details from job listings.

        Args:
            jobs (list): List of job elements from BeautifulSoup.

        Returns:
            list: List of job dictionaries containing job details.
        """
        job_list = []

        for job in jobs:
            published_date_tag = job.find('span', class_='sim-posted')
            published_date = published_date_tag.get_text(strip=True) if published_date_tag else 'N/A'

            company_tag = job.find('h3', class_='joblist-comp-name')
            company_name = re.sub(r'\s+', ' ', company_tag.get_text(strip=True)) if company_tag else 'N/A'

            more_info_tag = job.find('a', class_='posoverlay_srp')
            more_info = more_info_tag.get('href', 'N/A').strip() if more_info_tag else 'N/A'

            # Process only jobs with recent postings (e.g., 'few' days ago)
            if 'few' in published_date.lower():
                job_details = JobDetails.get_job_details(more_info)
                job_title = job_details.get("job_title", "N/A")
                
                job_data = {
                    "Job Title": job_title,
                    "Company Name": company_name,
                    "Required Skills": ', '.join(job_details.get("skills", [])),
                    "Experience": job_details.get("experience", "N/A"),
                    "Salary": job_details.get("salary", "N/A"),
                    "Location": job_details.get("location", "N/A"),
                    "More Info": more_info
                }
                job_list.append(job_data)

        return job_list


def main():
    """Main function to run the job scraper and save results to a CSV file."""
    base_url = (
        "https://www.timesjobs.com/candidate/job-search.html?"
        "searchType=personalizedSearch&from=submit&searchTextSrc=ft&"
        "searchTextText=%22Software+Developer%22&txtKeywords=%22Software+Developer%22%2C&txtLocation="
    )

    all_jobs = []

    for page in range(1, 121):
        url = f"{base_url}&sequence={page}"
        print(f"Scraping page {page} with URL: {url}")

        scraper = JobScraper(url)
        job_list = scraper.fetch_jobs()
        
        all_jobs.extend(job_list)

    # Save all jobs to CSV
    save_to_csv(all_jobs, filename="scraped_jobs2.csv", overwrite=True)


if __name__ == "__main__":
    main()