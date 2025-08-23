from job_description_handling import job_description_phrase_scraper
from resume_handling import resume_phrase_scraper
from resume_handling import resume_ranker
from rewriteparagraph import new_experience
from nsquare import exportkeywords

"""
format_type = input("Select format type: [plain text/HTML]: ")
if format_type == "HTML":
    web_file = input("Enter website name: ")
elif format_type == "plain text":
    plain_text = input("Enter plain text: ")
    phrases = job_description_phrase_scraper(plain_text)
"""
resume_experiences = []
continue_input = True
while continue_input:
    resume_experience_input = input("Enter resume experiences or END to stop: ")
    if resume_experience_input == "END":
        continue_input = False
        break
    else:
        resume_experiences.append(resume_experience_input)
#while resume_experience_input != "END": the while here stopped one experience from getting recorded
#    resume_experiences.append(resume_experience_input)



"""
resume_phrases = [resume_phrase_scraper(resume_experience) for resume_experience in resume_experiences]
resume_ranks = [resume_ranker(resume_phrase) for resume_phrase in resume_phrases]
sorted_experiences = [experience for _, experience in sorted(zip(resume_ranks, resume_experiences))]
for i,experience in enumerate(sorted_experiences):
    print(1+i, ". ", experience, sep="")
"""
occupation = input("List job title here: ")
new_experiences = []
for i in resume_experiences:
    print(i)
    new_resume_experience = new_experience(i,occupation)
    new_experiences.append(new_resume_experience)

print(new_experiences)
