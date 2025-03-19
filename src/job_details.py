import requests
from bs4 import BeautifulSoup
import re


class JobDetails:
    """
    Extracts detailed job information from a given job listing page.
    """

    @staticmethod
    def get_job_details(job_url):
        """
        Fetches detailed job information such as job title, experience, salary,
        location, and required skills from a job listing page.

        :param job_url: The URL of the job listing.
        :return: Dictionary containing job details.
        """
        if not job_url or job_url == 'N/A':
            return {}

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        try:
            response = requests.get(job_url, headers=headers)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to fetch job details from {job_url}. Error: {e}")
            return {}

        job_soup = BeautifulSoup(response.text, 'lxml')

        # Extract job title
        job_title_tag = job_soup.find('h1', class_='jd-job-title')
        job_title = job_title_tag.get_text(strip=True) if job_title_tag else 'N/A'

        # Extract experience and salary details
        details_list = job_soup.find('ul', class_='top-jd-dtl d-flex mt-8')
        experience, salary = 'N/A', 'N/A'
        if details_list:
            list_items = details_list.find_all('li')
            for li in list_items:
                text = " ".join(li.stripped_strings)
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                if li.find('i', class_='srp-icons experience'):
                    experience = text
                elif li.find('i', class_='srp-icons salary'):
                    salary = text

        # Extract location
        location_tag = job_soup.find('span', class_='job-location-trunicate')
        location = location_tag.get_text(strip=True) if location_tag else 'N/A'

        # Extract required skills
        skill_tags = job_soup.find_all('span', class_='jd-skill-tag')
        skills = [re.sub(r'\s+', ' ', skill.get_text(strip=True))
                  for skill in skill_tags if skill.get_text(strip=True)]

        return {
            "job_title": job_title,
            "experience": experience,
            "salary": salary,
            "location": location,
            "skills": skills
        }