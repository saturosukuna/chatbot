from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins (you can customize this for specific origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this list for specific origins, e.g., ["https://example.com"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods, adjust as necessary
    allow_headers=["*"],  # Allow all headers, adjust as necessary
)

class Query(BaseModel):
    question: str

class Response(BaseModel):
    answer: str

qa_dataset = [
  {
    "question": "What is your name?",
    "answer": "My name is Rajesh."
  },
    {
      "question": "Where are you from?",
      "answer": "I am from Tamil Nadu, India."
    },
    {
      "question": "What are you studying?",
      "answer": "I am completing my Bachelor's in Information Technology in 2025 from Annamalai University."
    },
    {
      "question": "What skills do you have?",
      "answer": "I have strong logical thinking and knowledge in deep learning, blockchain, networking, web development, software development, and animations."
    },
    {
      "question": "What languages do you know?",
      "answer": "I know English and Tamil with decent proficiency."
    },
    {
      "question": "What is your GitHub username?",
      "answer": "My GitHub username is 'saturosukuna'."
    },
    {
      "question": "What is your portfolio website link?",
      "answer": "You can view my portfolio at 'https://saturosukuna.github.io/portfolio'."
    },
    {
      "question": "What is your goal?",
      "answer": "I aim to become a MERN stack developer and am passionate about working on innovative projects."
    },
    {
      "question": "Tell me about your blockchain project.",
      "answer": "I am working on an academic blockchain project where every participant updates their own information. All updates and changes are visible to everyone in the network, with no roles or admins. It uses technologies like Truffle, Ganache, Metamask, Web3, Ether, Solidity, and the MERN stack."
    },
    {
      "question": "What are you currently building?",
      "answer": "I am creating a file-sharing application for LAN (WiFi or hotspot) with both an EXE for laptops and an APK for mobile devices."
    },
    {
      "question": "What technologies do you use for your frontend?",
      "answer": "I use Vite with React and JavaScript for my frontend development."
    },
    {
      "question": "What is your system configuration?",
      "answer": "I have an RTX 2050 GPU and an Intel i5-11260H processor."
    },
  {
    "question": "When were you born?",
    "answer": "I was born on 08-06-2004."
  },
  {
    "question": "What is your residential address?",
    "answer": "My residential address is 20, Thiruppanazhuar Street, Srimushnam, Cuddalore District, Tamil Nadu, 608703, India."
  },
  {
    "question": "What is your native address?",
    "answer": "My native address is 1003, Melatheru, Vadugarpalayam, Ariyalur District, Tamil Nadu, 621803, India."
  },
  {
    "question": "Tell me about your family.",
    "answer": "I come from a loving family. My father, Ravi, is a farmer. My mother, Rajalakshmi, is a tailor and now a housewife. My elder sister, Ilakkiya, is a physiotherapist."
  },
  {
    "question": "When was your sister born?",
    "answer": "My sister, Ilakkiya, was born on 20-08-2002."
  },
  {
    "question": "Where did you study for your primary education?",
    "answer": "I attended Devi Nursery and Primary School, Virudhachalam, for my kindergarten to primary education."
  },
  {
    "question": "What was your personality like during your primary school years?",
    "answer": "I was introverted during my primary school years, excelling academically and often ranking among the top students."
  },
  {
    "question": "Where did you study from 6th to 12th grade?",
    "answer": "I studied from 6th to 12th grade at DVC Higher Secondary School, Srimushnam."
  },
  {
    "question": "How did you perform in your 10th grade?",
    "answer": "In my 10th grade, I achieved excellent marks: Maths: 95, Science: 91, Social Studies: 96, Tamil: 92, English: 86."
  },
  {
    "question": "What sports did you participate in during school?",
    "answer": "I actively participated in kho-kho, football, and throwball during my school years."
  },
  {
    "question": "What awards or certificates did you receive in school?",
    "answer": "I received certificates for full attendance and second place in a 5th-grade GK quiz competition."
  },
  {
    "question": "What happened during your 12th grade?",
    "answer": "My 12th-grade exams were disrupted due to the COVID-19 pandemic, and I scored 539."
  },
  {
    "question": "What are you currently studying?",
    "answer": "I am pursuing my Bachelor's in Information Technology at Annamalai University."
  },
  {
    "question": "What is your current OGPA?",
    "answer": "My current OGPA is 8.66."
  },
  {
    "question": "Do you have any academic arrears?",
    "answer": "No, I have no academic arrears."
  },
  {
    "question": "What programming languages do you know?",
    "answer": "I am proficient in Python, Java, and C++."
  },
  {
    "question": "What technologies are you skilled in?",
    "answer": "I am skilled in the MERN stack, Flask, Vite, MongoDB, MySQL, and have explored R, Blender, Kotlin, and Django."
  },
  {
    "question": "Tell me about your blockchain project.",
    "answer": "I developed a Decentralized Identity Management System using blockchain, which securely manages identities, ensuring transparency and data integrity. It includes roles like Admin, Teacher, and Student."
  },
  {
    "question": "What is your GitHub username?",
    "answer": "My GitHub username is 'saturosukuna'."
  },
  {
    "question": "What is your portfolio website link?",
    "answer": "You can view my portfolio at 'https://saturosukuna.github.io/portfolio'."
  },
  {
    "question": "What devices do you own?",
    "answer": "I own a Poco F6 mobile and an ASUS TUF Gaming F15 laptop with an RTX 2050 GPU and Intel i5-11260H processor."
  },
  {
    "question": "What are your hobbies?",
    "answer": "During my childhood, I enjoyed watching TV, playing games with friends, and excelling in studies."
  },
  {
    "question": "What is your goal?",
    "answer": "I aim to become a MERN stack developer and contribute meaningfully to software development."
  },
  
]


try:
    qa_pipeline = pipeline("question-answering")
    context = "\n".join([f"Q: {item['question']} A: {item['answer']}" for item in qa_dataset])
except Exception as init_error:
    raise RuntimeError(f"Error initializing the QA pipeline: {str(init_error)}")

@app.post("/chat", response_model=Response)
async def chat(query: Query):
    try:
        result = qa_pipeline(question=query.question, context=context)
        if result and result.get("answer"):
            return Response(answer=result["answer"])
        raise HTTPException(status_code=404, detail="Answer not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Chatbot!"}