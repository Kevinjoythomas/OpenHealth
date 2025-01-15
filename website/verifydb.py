import os
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma"

def verify_documents_in_chroma():
    if not os.path.exists(CHROMA_PATH):
        print(f"❌ No Chroma database found at {CHROMA_PATH}. Please ensure the database is created.")
        return

    db = Chroma(persist_directory=CHROMA_PATH)

    existing_items = db.get(include=[])
    if existing_items["ids"]:
        print(f"✅ {len(existing_items['ids'])} documents found in the Chroma database:")
        pdf_names = set()  
        for doc_id in existing_items["ids"]:
            pdf_name = doc_id.split("\\")[1].split(".pdf")[0]
            pdf_names.add(pdf_name)
        
        for pdf_name in pdf_names:
            print(f"  - PDF Name: {pdf_name}")
    else:
        print("👉 No documents found in the Chroma database.")

def main():
    verify_documents_in_chroma()

if __name__ == "__main__":
    main()

'''
Output should be something like this:
✅ 1823 documents found in the Chroma database:
  - PDF Name: legPain
  - PDF Name: 14. Bone Cancer (Article) Author Oncology Nurse Advisor
  - PDF Name: eyeHealth
  - PDF Name: tonsillitis
  - PDF Name: stress
  - PDF Name: neckPain(full)
  - PDF Name: pain
  - PDF Name: 2. Cancer Author Tata Memorial Centre
  - PDF Name: heartDisease
  - PDF Name: fatigue
  - PDF Name: muscleSoreness
  - PDF Name: 11. Cancer Prevention and Control in India Author Cherian Varghese
  - PDF Name: tuberculosis
  - PDF Name: breathingPatterns
  - PDF Name: childrenHeadaches
  - PDF Name: footPain
  - PDF Name: elderlyHeadaches
  - PDF Name: backPain
  - PDF Name: 5. Skin Cancer Author Lauren Queen
  - PDF Name: ArmAndNeckPain
  - PDF Name: chestPain(full)
  '''
  