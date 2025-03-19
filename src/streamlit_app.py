import streamlit as st
import re
from query_faiss import search_jobs  # Import FAISS + Gemini function

# Streamlit UI Setup
st.title("🔍 AI-Powered Job Search")
st.write("Enter a job query below and get AI-refined job listings!")

#  User Input for Job Query
query = st.text_input("💼 Job Search Query:", "")

def is_valid_query(query):
    """
    Validates the user query.
    - Rejects queries with only numbers.
    - Rejects queries with random characters (garbage input).
    - Ensures query contains meaningful words.

    :param query: User input string
    :return: True if valid, False otherwise
    """
    if query.isdigit():  # Only numbers
        return False
    if not re.search(r"[a-zA-Z]", query):  # No meaningful words
        return False
    if len(query) < 3:  # Too short to be meaningful
        return False
    return True

if st.button("Search Jobs"):
    if query:
        if not is_valid_query(query):
            st.warning("⚠️ Please enter a valid job query. Avoid using only numbers or random characters.")
        else:
            st.info("🔄 Searching for the best job matches...")

            # Call FAISS + Gemini function to retrieve AI-refined job results
            refined_results = search_jobs(query, k=20)  
            if refined_results:
                st.subheader("🎯 AI-Refined Job Listings:")

                # Split the AI response into individual job postings
                jobs = refined_results.split("\n\n")  # Assumes jobs are separated by double newlines
                
                for idx, job in enumerate(jobs, 1):
                    lines = job.splitlines()
                    
                    if len(lines) > 1:
                        company_line = lines[0].strip()  # First line is company name
                        details = "\n".join(lines[1:])  # Remaining job details
                        
                        # Display company name in bold + "is hiring!"
                        st.markdown(f"<h3><b>{idx}. {company_line} </b></h3>", unsafe_allow_html=True)
                        
                        # Display job details below with spacing
                        st.write(details)

                    # Add a horizontal divider for clarity
                    st.markdown("<hr style='border: 1px solid #ccc; margin: 20px 0;'>", unsafe_allow_html=True)
                
            else:
                st.warning("❌ No relevant jobs found.")
    else:
        st.warning("⚠️ Please enter a job query.")
