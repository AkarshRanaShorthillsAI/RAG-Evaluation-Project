import csv
import os

def save_to_csv(job_list, filename="scraped_jobs.csv", overwrite=False):
    """
    Save job listings to a CSV file.

    Args:
        job_list (list): List of job dictionaries to save.
        filename (str, optional): Name of the CSV file. Defaults to "scraped_jobs.csv".
        overwrite (bool, optional): If True, creates a new file instead of appending. Defaults to False.
    """
    if not job_list:
        print("⚠️ No job data to save.")
        return

    # Determine file mode: 'w' (overwrite) or 'a' (append)
    mode = 'w' if overwrite or not os.path.isfile(filename) else 'a'

    with open(filename, mode=mode, newline="", encoding="utf-8") as file:
        fieldnames = [
            "Job Title", "Company Name", "Required Skills", "Experience",
            "Salary", "Location", "More Info"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write headers only in 'w' mode (new file)
        if mode == 'w':
            writer.writeheader()

        # Write job data
        writer.writerows(job_list)

    print(f"✅ Data successfully saved to {filename}")
