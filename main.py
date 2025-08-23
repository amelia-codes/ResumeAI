from job_description_handling import job_description_phrase_scraper
from resume_handling import resume_phrase_scraper
from resume_handling import resume_ranker

format_type = input("Select format type: [plain text/HTML]: ")
if format_type == "HTML":
    web_file = input("Enter website name:")
elif format_type == "plain text":
    plain_text = input("Enter plain text:")
    phrases = job_description_phrase_scraper(plain_text)

resume_experiences = []
resume_experience_input = input("Enter resume experiences or END to stop:")
while resume_experience_input != "END":
    resume_experiences.append(resume_experience_input)
    resume_experience_input = input("Enter resume experiences or END to stop:")

resume_phrases = [resume_phrase_scraper(resume_experience) for resume_experience in resume_experiences]
resume_ranks = [resume_ranker(resume_phrase) for resume_phrase in resume_phrases]
sorted_experiences = [experience for _, experience in sorted(zip(resume_ranks, resume_experiences))]
for i,experience in enumerate(sorted_experiences):
    print(1+i, ". ", experience, sep="")