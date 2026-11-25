"""YojanaSetu Voice Assistant pipeline.

Provides:
- Dynamic SQLite schema inspection (tables, columns, sample categories/beneficiaries).
- Multilingual input handling for Hindi, Marathi, Telugu, Tamil, Kannada,
  Bengali, Hinglish, and English.
- Entity extraction (state, age, occupation/beneficiary type, and need).
- Natural-language-to-SQLite pipeline using Gemini or OpenAI (via `.env` keys)
  with a deterministic multilingual keyword + fuzzy fallback when no API key
  is configured.
- Strict 4-field response synthesis in the user's exact language:
  Scheme Name, Eligibility, Documents Needed, and Official Website link.
- Audio transcription (Gemini / OpenAI) and Text-to-Speech (`gTTS`) helpers.
"""

from __future__ import annotations

import difflib
import io
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

SUPPORTED_LANGUAGES: list[str] = [
    "Auto-detect",
    "Hindi",
    "Marathi",
    "Telugu",
    "Tamil",
    "Kannada",
    "Bengali",
    "Hinglish",
    "English",
]

TTS_LANGUAGE_CODES: dict[str, str] = {
    "Hindi": "hi",
    "Marathi": "mr",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Bengali": "bn",
    "Hinglish": "hi",
    "English": "en",
}

FIELD_LABELS: dict[str, dict[str, str]] = {
    "English": {
        "scheme_name": "Scheme Name",
        "eligibility": "Eligibility",
        "documents_needed": "Documents Needed",
        "official_website": "Official Website",
    },
    "Hindi": {
        "scheme_name": "योजना का नाम",
        "eligibility": "पात्रता",
        "documents_needed": "आवश्यक दस्तावेज़",
        "official_website": "आधिकारिक वेबसाइट",
    },
    "Marathi": {
        "scheme_name": "योजनेचे नाव",
        "eligibility": "पात्रता",
        "documents_needed": "आवश्यक कागदपत्रे",
        "official_website": "अधिकृत वेबसाइट",
    },
    "Telugu": {
        "scheme_name": "పథకం పేరు",
        "eligibility": "అర్హత",
        "documents_needed": "అవసరమైన పత్రాలు",
        "official_website": "అధికారిక వెబ్‌సైట్",
    },
    "Tamil": {
        "scheme_name": "திட்டத்தின் பெயர்",
        "eligibility": "தகுதி",
        "documents_needed": "தேவையான ஆவணங்கள்",
        "official_website": "அதிகாரப்பூர்வ இணையதளம்",
    },
    "Kannada": {
        "scheme_name": "ಯೋಜನೆಯ ಹೆಸರು",
        "eligibility": "ಅರ್ಹತೆ",
        "documents_needed": "ಅಗತ್ಯ ದಾಖಲೆಗಳು",
        "official_website": "ಅಧಿಕೃತ ವೆಬ್‌ಸೈಟ್",
    },
    "Bengali": {
        "scheme_name": "প্রকল্পের নাম",
        "eligibility": "যোগ্যতা",
        "documents_needed": "প্রয়োজনীয় নথিপত্র",
        "official_website": "অফিশিয়াল ওয়েবসাইট",
    },
    "Hinglish": {
        "scheme_name": "Yojana ka Naam",
        "eligibility": "Patrata (Eligibility)",
        "documents_needed": "Zaroori Documents",
        "official_website": "Official Website",
    },
}

# Multilingual state normalization -> canonical English state names
STATE_SYNONYMS: dict[str, list[str]] = {
    "Maharashtra": ["maharashtra", "महाराष्ट्र", "maharastra", "mumbai", "pune", "nagpur"],
    "Madhya Pradesh": ["madhya pradesh", "मध्य प्रदेश", "mp", "bhopal", "indore"],
    "Telangana": ["telangana", "तेलंगाना", "తెలంగాణ", "hyderabad", "warangal"],
    "Andhra Pradesh": ["andhra pradesh", "andhra", "आंध्र प्रदेश", "ఆంధ్రప్రదేశ్", "vijayawada", "visakhapatnam"],
    "West Bengal": ["west bengal", "bengal", "पश्चिम बंगाल", "পশ্চিমবঙ্গ", "kolkata"],
    "Tamil Nadu": ["tamil nadu", "tamilnadu", "तमिलनाडु", "தமிழ்நாடு", "தமிழ்நாட்", "chennai", "madurai"],
    "Karnataka": ["karnataka", "कर्नाटक", "ಕರ್ನಾಟಕ", "bengaluru", "bangalore", "mysuru"],
    "Uttar Pradesh": ["uttar pradesh", "up", "उत्तर प्रदेश", "lucknow", "varanasi"],
    "Bihar": ["bihar", "बिहार", "patna"],
    "Rajasthan": ["rajasthan", "राजस्थान", "jaipur"],
    "Gujarat": ["gujarat", "गुजरात", "ahmedabad", "surat"],
    "Kerala": ["kerala", "केरल", "kochi", "thiruvananthapuram"],
    "Odisha": ["odisha", "orissa", "ओडिशा", "bhubaneswar"],
    "Punjab": ["punjab", "पंजाब", "chandigarh", "ludhiana"],
    "Haryana": ["haryana", "हरियाणा", "gurugram"],
}

# Multilingual keywords mapped to canonical English search/scoring concepts
CONCEPT_KEYWORDS: dict[str, dict[str, Any]] = {
    "farmer": {
        "terms": [
            "farmer", "farmers", "agriculture", "farming", "crop", "kisan", "krishi",
            "किसान", "खेती", "कृषि", "फसल", "जमीन",
            "शेतकरी", "शेती", "पीक",
            "రైతు", "వ్యవసాయం", "పంట",
            "விவசாயி", "விவசாயம்", "பயிர்",
            "ರೈತ", "ಕೃಷಿ", "ಬೆಳೆ",
            "কৃষক", "চাষি", "চাষ", "কৃষি", "ফসল",
        ],
        "categories": ["Agriculture", "Rural Development", "Finance", "Insurance"],
        "beneficiaries": ["Farmer", "Farmers", "Small and marginal farmers", "Organic farmers"],
        "occupation": "Farmer",
    },
    "student": {
        "terms": [
            "student", "students", "education", "scholarship", "school", "college", "study",
            "padhai", "chhatra", "vidyarthi", "shiksha",
            "छात्र", "छात्रा", "विद्यार्थी", "पढ़ाई", "शिक्षा", "छात्रवृत्ति", "स्कूल", "कॉलेज",
            "विद्यार्थिनी", "शिक्षण", "शिष्यवृत्ती", "शाळा",
            "విద్యార్థి", "చదువు", "విద్య", "స్కాలర్‌షిప్",
            "மாணவர்", "மாணவி", "கல்வி", "பள்ளி", "உதவித்தொகை",
            "ವಿದ್ಯಾರ್ಥಿ", "ಶಿಕ್ಷಣ", "ವಿದ್ಯಾರ್ಥಿವೇತನ", "ಶಾಲೆ",
            "ছাত্র", "ছাত্রী", "শিক্ষা", "বৃত্তি", "পড়াশোনা", "স্কুল",
        ],
        "categories": ["Education", "Skill Development", "Innovation"],
        "beneficiaries": ["Student", "Students", "School Children", "Girl Students"],
        "occupation": "Student",
    },
    "women": {
        "terms": [
            "woman", "women", "girl", "girls", "daughter", "mahila", "beti", "pregnant", "maternity", "ladli",
            "महिला", "महिलाओं", "लड़की", "बेटी", "गर्भवती", "मातृ", "कन्या",
            "महिलांसाठी", "मुलगी", "मुली", "बहीण", "लाडकी", "गरोदर",
            "మహిళ", "అమ్మాయి", "గర్భిణి", "కూతురు",
            "பெண்", "பெண்கள்", "மகள்", "கர்ப்பிணி",
            "ಮಹಿಳೆ", "ಹೆಣ್ಣು", "ಮಗಳು", "ಗರ್ಭಿಣಿ",
            "মহিলা", "নারী", "মেয়ে", "কন্যা", "গর্ভবতী",
        ],
        "categories": ["Women and Child Development", "Health", "Social Welfare", "Finance"],
        "beneficiaries": ["Women", "Girl Child", "Girl Students", "Pregnant women"],
        "occupation": "Women / Girl Child",
    },
    "health": {
        "terms": [
            "health", "hospital", "medical", "treatment", "insurance", "doctor", "medicine", "illness",
            "swasthya", "ilaj", "bima", "aspatal", "dawai", "aarogya",
            "स्वास्थ्य", "अस्पताल", "इलाज", "बीमा", "चिकित्सा", "दवाई", "आरोग्य",
            "आरोग्य", "रुग्णालय", "उपचार", "विमा",
            "ఆరోగ్యం", "ఆసుపత్రి", "వైద్యం", "బీమా",
            "சுகாதாரம்", "மருத்துவமனை", "சிகிச்சை", "காப்பீடு",
            "ಆರೋಗ್ಯ", "ಆಸ್ಪತ್ರೆ", "ಚಿಕಿತ್ಸೆ", "ವಿಮೆ",
            "স্বাস্থ্য", "হাসপাতাল", "চিকিৎসা", "বীমা",
        ],
        "categories": ["Health", "Insurance"],
        "beneficiaries": ["BPL Families", "Poor and Vulnerable Families", "Citizens"],
        "need": "Healthcare & Health Insurance",
    },
    "housing": {
        "terms": [
            "house", "housing", "home", "awas", "ghar", "shelter", "makkan",
            "घर", "आवास", "मकान", "रहने",
            "घरकुल", "निवास",
            "ఇల్లు", "గృహ",
            "வீடு", "குடியிருப்பு",
            "ಮನೆ", "ವಸತಿ",
            "বাড়ি", "গৃহ", "বাসস্থান",
        ],
        "categories": ["Housing", "Rural Development", "Urban Development"],
        "beneficiaries": ["Rural Households", "Urban and rural poor", "Economically Weaker Section"],
        "need": "Housing / Home Construction",
    },
    "business_loan": {
        "terms": [
            "business", "loan", "startup", "entrepreneur", "vendor", "shop", "artisan", "craft", "msme",
            "mudra", "karz", "udyam", "vyapar", "dukaan", "rozgar", "self-employed",
            "व्यापार", "लोन", "ऋण", "कर्ज", "दुकान", "उद्यम", "व्यवसाय", "रोजगार", "कारीगर",
            "व्यवसाय", "कर्ज", "उद्योग",
            "వ్యాపారం", "రుణం", "లోన్",
            "தொழில்", "கடன்", "வணிகம்",
            "ವ್ಯಾಪಾರ", "ಸಾಲ", "ಉದ್ಯಮ",
            "ব্যবসা", "ঋণ", "লোন", "উদ্যোগ",
        ],
        "categories": ["Business", "Entrepreneurship", "Finance", "Banking and Finance", "Employment"],
        "beneficiaries": ["Entrepreneurs", "Small business owners", "Street vendors", "Traditional Artisans"],
        "need": "Business Loan / Self-Employment",
    },
    "employment_skill": {
        "terms": [
            "job", "jobs", "employment", "work", "skill", "training", "apprentice", "unemployed", "mazdoor",
            "naukri", "kaam", "kaushal", "prashikshan", "berozgar",
            "नौकरी", "काम", "रोजगार", "कौशल", "प्रशिक्षण", "बेरोजगार", "मजदूर",
            "नोकरी", "काम", "कौशल्य", "प्रशिक्षण", "बेरोजगार",
            "ఉద్యోగం", "పని", "నైపుణ్యం", "శిక్షణ",
            "வேலை", "திறன்", "பயிற்சி", "வேலைவாய்ப்பு",
            "ಕೆಲಸ", "ಉದ್ಯೋಗ", "ಕೌಶಲ್ಯ", "ತರಬೇತಿ",
            "চাকরি", "কাজ", "কর্মসংস্থান", "দক্ষতা", "প্রশিক্ষণ",
        ],
        "categories": ["Employment", "Skill Development", "Urban Employment", "Rural Development"],
        "beneficiaries": ["Youth", "Unemployed youth", "Unorganized workers", "Rural households"],
        "need": "Employment / Skill Training",
    },
    "pension_elderly": {
        "terms": [
            "pension", "elderly", "senior", "old age", "widow", "disabled", "retirement",
            "buzurg", "budhapa", "vidhwa", "divyang",
            "पेंशन", "बुजुर्ग", "वृद्ध", "विधवा", "दिव्यांग",
            "पेन्शन", "ज्येष्ठ", "वृद्धापकाळ", "विधवा",
            "పెన్షన్", "వృద్ధాప్య", "వితంతు",
            "ஓய்வூதியம்", "முதியோர்", "விதவை",
            "ಪಿಂಚಣಿ", "ಹಿರಿಯ", "ವೃದ್ಧಾಪ್ಯ", "ವಿಧವೆ",
            "পেনশন", "বয়স্ক", "বিধবা", "ভাতা",
        ],
        "categories": ["Social Security", "Social Welfare", "Finance", "Insurance"],
        "beneficiaries": ["Elderly, widows, disabled", "Unorganized Sector Workers", "Working Population"],
        "need": "Pension / Social Security",
    },
    "energy_gas": {
        "terms": [
            "gas", "lpg", "cylinder", "electricity", "power", "solar", "light", "bijli", "ujjwala",
            "गैस", "सिलेंडर", "बिजली", "सौर", "रोशनी",
            "गॅस", "वीज", "सौर",
            "గ్యాస్", "విద్యుత్", "కరెంట్",
            "எரிவாயு", "மின்சாரம்",
            "ಗ್ಯಾಸ್", "ವಿದ್ಯುತ್", "ಸೌರ",
            "গ্যাস", "বিদ্যুৎ",
        ],
        "categories": ["Energy", "Social Welfare", "Environment"],
        "beneficiaries": ["Women from BPL households", "Rural Households", "Un-electrified households"],
        "need": "Clean Cooking Fuel / Electricity / Solar",
    },
    "water_sanitation": {
        "terms": [
            "water", "drinking water", "tap", "toilet", "sanitation", "cleanliness", "pani", "jal", "shauchalay",
            "पानी", "जल", "शौचालय", "स्वच्छता",
            "पाणी", "स्वच्छता", "शौचालय",
            "నీరు", "మంచినీరు", "మరుగుదొడ్డి",
            "தண்ணீர்", "குடிநீர்", "கழிப்பறை",
            "ನೀರು", "ಕುಡಿಯುವ ನೀರು", "ಶೌಚಾಲಯ",
            "জল", "পানীয় জল", "শৌচাগার",
        ],
        "categories": ["Water and Sanitation", "Water", "Sanitation", "Rural Development"],
        "beneficiaries": ["Rural Households", "Households", "All citizens"],
        "need": "Clean Water / Sanitation",
    },
    "food_ration": {
        "terms": [
            "food", "ration", "grain", "nutrition", "anaj", "rashan", "khana",
            "राशन", "अनाज", "खाना", "पोषण", "खाद्य",
            "रेशन", "अन्न", "धान्य",
            "రేషన్", "ఆహారం", "బియ్యం",
            "ரேஷன்", "உணவு", "அரிசி",
            "ರೇಷನ್", "ಆಹಾರ", "ಅಕ್ಕಿ",
            "রেশন", "খাদ্য", "চাল",
        ],
        "categories": ["Food Security", "Social Welfare"],
        "beneficiaries": ["BPL and AAY Families", "School Children", "Poor and Vulnerable Families"],
        "need": "Food Security / Ration",
    },
}

# Canonical document extraction patterns from Hindi/English DB answers
DOCUMENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("Aadhaar Card", ["आधार", "aadhaar", "aadhar"]),
    ("Bank Account / Passbook", ["बैंक", "पासबुक", "bank", "passbook", "ifsc"]),
    ("Land / Property Records (Khatauni / 7/12)", ["जमीन के कागज", "जमीन का कागज", "भूमि अभिलेख", "भूमि रिकॉर्ड", "खतौनी", "जमाबंदी", "land record", "patta", "khatauni"]),
    ("Income Certificate", ["आय प्रमाण", "income certificate"]),
    ("Residence / Domicile Certificate", ["निवास प्रमाण", "domicile", "residence certificate", "ration card", "राशन कार्ड"]),
    ("Caste Certificate (if applicable)", ["जाति प्रमाण", "caste certificate", "sc/st"]),
    ("Passport-size Photographs", ["फोटो", "तस्वीर", "photograph", "photo"]),
    ("Active Mobile Number", ["मोबाइल नंबर", "mobile number", "phone number"]),
    ("Age / Birth Certificate", ["जन्म प्रमाण", "आयु प्रमाण", "birth certificate", "age proof"]),
    ("Educational / School Certificates", ["शैक्षणिक", "मार्कशीट", "स्कूल", "marksheet", "school", "college id"]),
    ("BPL / Ration Card", ["बीपीएल", "bpl", "राशन कार्ड", "ration card", "aay"]),
    ("PAN Card / Business Registration", ["पैन कार्ड", "pan card", "udyam", "gst", "dpiit"]),
]

DOCUMENT_TRANSLATIONS: dict[str, dict[str, str]] = {
    "Aadhaar Card": {
        "Hindi": "आधार कार्ड",
        "Marathi": "आधार कार्ड",
        "Telugu": "ఆధార్ కార్డు",
        "Tamil": "ஆதார் அட்டை",
        "Kannada": "ಆಧಾರ್ ಕಾರ್ಡ್",
        "Bengali": "আধার কার্ড",
        "Hinglish": "Aadhaar Card",
        "English": "Aadhaar Card",
    },
    "Bank Account / Passbook": {
        "Hindi": "बैंक खाता / पासबुक",
        "Marathi": "बँक खाते / पासबुक",
        "Telugu": "బ్యాంకు ఖాతా / పాస్‌బుక్",
        "Tamil": "வங்கி கணக்கு / பாஸ்புக்",
        "Kannada": "ಬ್ಯಾಂಕ್ ಖಾತೆ / ಪಾಸ್‌ಬುಕ್",
        "Bengali": "ব্যাংক অ্যাকাউন্ট / পাসবুক",
        "Hinglish": "Bank Account / Passbook",
        "English": "Bank Account / Passbook",
    },
    "Land / Property Records (Khatauni / 7/12)": {
        "Hindi": "भूमि / जमीन के कागजात (खतौनी)",
        "Marathi": "जमिनीचे कागदपत्र (७/१२ उतारा)",
        "Telugu": "భూమి పట్టా / పాస్‌బుక్ పత్రాలు",
        "Tamil": "நில ஆவணங்கள் (பட்டா / சிட்டா)",
        "Kannada": "ಭೂಮಿ ದಾಖಲೆಗಳು (ಪಹಣಿ / RTC)",
        "Bengali": "জমির কাগজপত্র (পর্চা / খতিয়ান)",
        "Hinglish": "Zameen ke Kagazat (Land Records)",
        "English": "Land / Property Records",
    },
    "Income Certificate": {
        "Hindi": "आय प्रमाण पत्र",
        "Marathi": "उत्पन्नाचा दाखला",
        "Telugu": "ఆదాయ ధృవీకరణ పత్రం",
        "Tamil": "வருமானச் சான்றிதழ்",
        "Kannada": "ಆದಾಯ ಪ್ರಮಾಣಪತ್ರ",
        "Bengali": "আয়ের শংসাপত্র",
        "Hinglish": "Income Certificate",
        "English": "Income Certificate",
    },
    "Residence / Domicile Certificate": {
        "Hindi": "निवास प्रमाण पत्र",
        "Marathi": "रहिवासी दाखला (Domicile)",
        "Telugu": "నివాస ధృవీకరణ పత్రం",
        "Tamil": "இருப்பிடச் சான்றிதழ்",
        "Kannada": "ನಿವಾಸಿ ಪ್ರಮಾಣಪತ್ರ",
        "Bengali": "বাসিন্দা শংসাপত্র",
        "Hinglish": "Niwas Praman Patra (Address Proof)",
        "English": "Residence / Address Proof",
    },
    "Caste Certificate (if applicable)": {
        "Hindi": "जाति प्रमाण पत्र (यदि लागू हो)",
        "Marathi": "जात प्रमाणपत्र (लागू असल्यास)",
        "Telugu": "కుల ధృవీకరణ పత్రం (అవసరమైతే)",
        "Tamil": "சாதிச் சான்றிதழ் (பொருந்தினால்)",
        "Kannada": "ಜಾತಿ ಪ್ರಮಾಣಪತ್ರ (ಅನ್ವಯಿಸಿದರೆ)",
        "Bengali": "জাতিগত শংসাপত্র (প্রযোজ্য হলে)",
        "Hinglish": "Caste Certificate (agar laagu ho)",
        "English": "Caste Certificate (if applicable)",
    },
    "Passport-size Photographs": {
        "Hindi": "पासपोर्ट साइज फोटो",
        "Marathi": "पासपोर्ट साईज फोटो",
        "Telugu": "పాస్‌పోర్ట్ సైజు ఫోటోలు",
        "Tamil": "பாஸ்போர்ட் அளவு புகைப்படங்கள்",
        "Kannada": "ಪಾಸ್‌ಪೋರ್ಟ್ ಅಳತೆಯ ಫೋಟೋಗಳು",
        "Bengali": "পাসপোর্ট সাইজ ছবি",
        "Hinglish": "Passport-size Photo",
        "English": "Passport-size Photographs",
    },
    "Active Mobile Number": {
        "Hindi": "आधार से लिंक मोबाइल नंबर",
        "Marathi": "आधारशी लिंक मोबाईल नंबर",
        "Telugu": "ఆధార్ లింక్ అయిన మొబైల్ నంబర్",
        "Tamil": "ஆதாருடன் இணைந்த மொபைல் எண்",
        "Kannada": "ಆಧಾರ್ ಲಿಂಕ್ ಆದ ಮೊಬೈಲ್ ಸಂಖ್ಯೆ",
        "Bengali": "আধার যুক্ত মোবাইল নম্বর",
        "Hinglish": "Aadhaar-linked Mobile Number",
        "English": "Aadhaar-linked Mobile Number",
    },
    "Age / Birth Certificate": {
        "Hindi": "आयु / जन्म प्रमाण पत्र",
        "Marathi": "वयाचा / जन्माचा दाखला",
        "Telugu": "వయస్సు / జనన ధృవీకరణ పత్రం",
        "Tamil": "வயது / பிறப்புச் சான்றிதழ்",
        "Kannada": "ವಯಸ್ಸು / ಜನನ ಪ್ರಮಾಣಪತ್ರ",
        "Bengali": "বয়স / জন্ম শংসাপত্র",
        "Hinglish": "Age / Birth Certificate",
        "English": "Age / Birth Certificate",
    },
    "Educational / School Certificates": {
        "Hindi": "शैक्षणिक प्रमाण पत्र / मार्कशीट",
        "Marathi": "शैक्षणिक प्रमाणपत्र / गुणपत्रक",
        "Telugu": "విద్యా అర్హత / మార్కుల పత్రాలు",
        "Tamil": "கல்விச் சான்றிதழ்கள் / மதிப்பெண் பட்டியல்",
        "Kannada": "ಶೈಕ್ಷಣಿಕ ಪ್ರಮಾಣಪತ್ರ / ಅಂಕಪಟ್ಟಿ",
        "Bengali": "শিক্ষাগত যোগ্যতার শংসাপত্র / মার্কশিট",
        "Hinglish": "School / College Marksheet",
        "English": "Educational / School Certificates",
    },
    "BPL / Ration Card": {
        "Hindi": "राशन कार्ड / BPL कार्ड",
        "Marathi": "रेशन कार्ड / BPL कार्ड",
        "Telugu": "రేషన్ కార్డు / BPL కార్డు",
        "Tamil": "ரேஷன் அட்டை / BPL அட்டை",
        "Kannada": "ರೇಷನ್ ಕಾರ್ಡ್ / BPL ಕಾರ್ಡ್",
        "Bengali": "রেশন কার্ড / BPL কার্ড",
        "Hinglish": "Ration Card / BPL Card",
        "English": "BPL / Ration Card",
    },
    "PAN Card / Business Registration": {
        "Hindi": "पैन कार्ड / व्यवसाय पंजीकरण",
        "Marathi": "पॅन कार्ड / व्यवसाय नोंदणी",
        "Telugu": "పాన్ కార్డు / వ్యాపార నమోదు",
        "Tamil": "பான் அட்டை / தொழில் பதிவு",
        "Kannada": "ಪ್ಯಾನ್ ಕಾರ್ಡ್ / ವ್ಯಾಪಾರ ನೋಂದಣಿ",
        "Bengali": "প্যান কার্ড / ব্যবসার নথি",
        "Hinglish": "PAN Card / Business Proof",
        "English": "PAN Card / Business Proof",
    },
}

MARATHI_DEVANAGARI_MARKERS = {
    "आहे", "आहेत", "नाही", "मी", "मला", "माझ्या", "माझे", "माझी", "साठी",
    "शेतकरी", "विद्यार्थी", "विद्यार्थिनी", "महाराष्ट्र", "हवी", "हवे", "पाहिजे",
    "कसे", "काय", "कोणत्या", "योजनेची", "योजनांची", "लाडकी", "बहीण", "घरकुल",
    "मिळेल", "सांगा", "दाखवा", "सरकारच्या", "तुम्ही", "आम्ही",
}

HINGLISH_MARKERS = {
    "mujhe", "mujhko", "mera", "meri", "mere", "main", "mai", "hum", "hamara",
    "kisaan", "chhatra", "vidyarthi", "mahila", "garib", "ghar", "awas",
    "yojana", "yojna", "sarkar", "sarkari", "chahiye", "chaahiye",
    "batao", "bataye", "bataiye", "kaunsi", "kaun", "kya", "kaise", "mil",
    "milega", "milegi", "karz", "paisa", "paise", "madad", "sahayata",
    "ke", "liye", "liya", "hai", "hain", "hoon", "hu", "ko", "se", "mein",
    "bima", "padhai", "naukri", "rozgar", "buzurg",
}

TELEGU_ROMAN_MARKERS = {"naaku", "nenu", "raitu", "rythu", "vidyarthi", "kavali", "cheppandi", "telangana", "andhra"}
TAMIL_ROMAN_MARKERS = {"enakku", "naan", "vivasayi", "maanavar", "thittam", "vendum", "sollunga", "tamilnadu"}
KANNADA_ROMAN_MARKERS = {"nanage", "naanu", "raitha", "yojane", "beku", "heli", "karnataka"}
BENGALI_ROMAN_MARKERS = {"amar", "ami", "chashi", "krishak", "chhatro", "prakalpa", "chai", "bolun", "bengal"}


def _get_api_keys() -> tuple[str | None, str | None]:
    """Return configured (gemini_key, openai_key) from environment."""
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if gemini_key and gemini_key.strip().startswith("your_"):
        gemini_key = None
    if openai_key and openai_key.strip().startswith("your_"):
        openai_key = None
    return (gemini_key.strip() if gemini_key else None, openai_key.strip() if openai_key else None)


def get_active_provider() -> str:
    """Determine active LLM provider: 'gemini', 'openai', or 'fallback'."""
    override = os.getenv("LLM_PROVIDER", "").strip().lower()
    gemini_key, openai_key = _get_api_keys()
    if override == "fallback":
        return "fallback"
    if override == "openai" and openai_key:
        return "openai"
    if override == "gemini" and gemini_key:
        return "gemini"
    if gemini_key:
        return "gemini"
    if openai_key:
        return "openai"
    return "fallback"


def _connect_readonly(db_path: str) -> sqlite3.Connection:
    """Open SQLite database in read-only URI mode when file exists."""
    resolved = Path(db_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"SQLite database not found: {resolved}")
    uri = f"file:{resolved.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def inspect_database_schema(db_path: str = "schemes.db") -> dict[str, Any]:
    """Dynamically inspect all tables, columns, row counts, and categorical samples.

    Queries `sqlite_master` and `PRAGMA table_info(...)` at runtime so SQL
    generation and fallback queries never rely on hardcoded table assumptions.
    """
    conn = _connect_readonly(db_path)
    try:
        cur = conn.cursor()
        table_rows = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        tables: dict[str, Any] = {}
        for row in table_rows:
            table_name = row["name"]
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name):
                continue
            cols_info = cur.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            columns: list[dict[str, Any]] = []
            samples: dict[str, list[Any]] = {}
            for col in cols_info:
                col_name = col["name"]
                col_type = col["type"] or "TEXT"
                columns.append({"name": col_name, "type": col_type})
                if any(
                    hint in col_name.lower()
                    for hint in ("category", "state", "central", "level", "type", "beneficiary", "difficulty")
                ):
                    distinct_vals = cur.execute(
                        f'SELECT DISTINCT "{col_name}" FROM "{table_name}" '
                        f'WHERE "{col_name}" IS NOT NULL AND TRIM("{col_name}") != "" LIMIT 25'
                    ).fetchall()
                    samples[col_name] = [r[0] for r in distinct_vals]

            row_count = cur.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            tables[table_name] = {
                "row_count": row_count,
                "columns": columns,
                "column_names": [c["name"] for c in columns],
                "categorical_samples": samples,
            }

        primary_table = _pick_primary_table(tables)
        col_map = _resolve_column_roles(tables[primary_table]["column_names"]) if primary_table else {}
        return {
            "db_path": str(Path(db_path).resolve()),
            "tables": tables,
            "primary_table": primary_table,
            "column_roles": col_map,
        }
    finally:
        conn.close()


def _pick_primary_table(tables: dict[str, Any]) -> str | None:
    """Pick the main schemes table dynamically based on column names."""
    if not tables:
        return None
    if "schemes" in tables:
        return "schemes"
    best_table = None
    best_score = -1
    for name, info in tables.items():
        cols = " ".join(info["column_names"]).lower()
        score = sum(1 for kw in ("scheme", "question", "answer", "website", "beneficiary", "category") if kw in cols)
        if score > best_score:
            best_score = score
            best_table = name
    return best_table or next(iter(tables))


def _resolve_column_roles(column_names: list[str]) -> dict[str, str]:
    """Map logical roles to actual discovered column names."""
    lower_map = {c.lower(): c for c in column_names}
    roles: dict[str, str] = {}
    candidates = {
        "scheme_name": ["scheme_name", "name", "title", "scheme"],
        "question": ["question", "query", "prompt"],
        "question_english": ["question_english", "question_en", "english_question"],
        "answer": ["answer", "details", "description", "answer_hindi"],
        "answer_english": ["answer_english", "answer_en", "english_answer", "description_en"],
        "category": ["category", "sector", "domain"],
        "state_or_central": ["state_or_central", "level", "govt_level", "scope", "state"],
        "target_beneficiary": ["target_beneficiary", "beneficiary", "beneficiaries", "audience"],
        "official_website": ["official_website", "website", "url", "link", "portal"],
        "launched_year": ["launched_year", "year", "launch_year"],
    }
    for role, opts in candidates.items():
        for opt in opts:
            if opt in lower_map:
                roles[role] = lower_map[opt]
                break
        if role not in roles:
            for col_l, col_orig in lower_map.items():
                if any(opt in col_l for opt in opts):
                    roles[role] = col_orig
                    break
    return roles


def format_schema_for_prompt(schema_info: dict[str, Any]) -> str:
    """Format inspected schema into a concise LLM-friendly summary."""
    lines: list[str] = []
    for table_name, meta in schema_info.get("tables", {}).items():
        col_desc = ", ".join(f'{c["name"]} ({c["type"]})' for c in meta["columns"])
        lines.append(f"Table `{table_name}` ({meta['row_count']} rows): {col_desc}")
        for col, vals in meta.get("categorical_samples", {}).items():
            preview = ", ".join(repr(v) for v in vals[:15])
            lines.append(f"  - Sample `{col}` values: [{preview}]")
    return "\n".join(lines)


def detect_language(query: str, preferred_language: str | None = None) -> str:
    """Detect user query language across the 8 supported Indian/English inputs."""
    if preferred_language and preferred_language != "Auto-detect" and preferred_language in FIELD_LABELS:
        return preferred_language

    text = (query or "").strip()
    if not text:
        return "English"

    # Script-based detection for regional scripts
    if re.search(r"[\u0C00-\u0C7F]", text):
        return "Telugu"
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "Tamil"
    if re.search(r"[\u0C80-\u0CFF]", text):
        return "Kannada"
    if re.search(r"[\u0980-\u09FF]", text):
        return "Bengali"
    if re.search(r"[\u0900-\u097F]", text):
        words = set(re.findall(r"[\u0900-\u097F]+", text))
        if words & MARATHI_DEVANAGARI_MARKERS or any(suf in text for suf in ("साठी", "च्या", "मध्ये", "लागते", "आहे")):
            return "Marathi"
        return "Hindi"

    # Latin script: distinguish Hinglish / Romanized regional / English
    tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
    if tokens & TELEGU_ROMAN_MARKERS and not (tokens & HINGLISH_MARKERS):
        return "Telugu"
    if tokens & TAMIL_ROMAN_MARKERS and not (tokens & HINGLISH_MARKERS):
        return "Tamil"
    if tokens & KANNADA_ROMAN_MARKERS and not (tokens & HINGLISH_MARKERS):
        return "Kannada"
    if tokens & BENGALI_ROMAN_MARKERS and not (tokens & HINGLISH_MARKERS):
        return "Bengali"
    if len(tokens & HINGLISH_MARKERS) >= 2 or (
        len(tokens & HINGLISH_MARKERS) >= 1 and any(w in tokens for w in ("mujhe", "chahiye", "batao", "bataye", "hoon", "yojana"))
    ):
        return "Hinglish"

    return "English"


def extract_query_entities(query: str, language: str | None = None) -> dict[str, Any]:
    """Extract state, age, occupation/beneficiary type, and need from a query."""
    detected_lang = detect_language(query, language)
    text_lower = (query or "").lower()

    # 1. State extraction
    detected_state: str | None = None
    for state_name, variants in STATE_SYNONYMS.items():
        for variant in variants:
            if len(variant) <= 3:
                if re.search(rf"\b{re.escape(variant)}\b", text_lower):
                    detected_state = state_name
                    break
            elif variant in text_lower:
                detected_state = state_name
                break
        if detected_state:
            break

    # 2. Age extraction (digits + multilingual age indicators)
    detected_age: int | None = None
    age_patterns = [
        r"(?:age|aged|yrs|years|year|साल|वर्ष|उम्र|वय|वर्षे|వయస్సు|సంవత్సరాలు|வயது|ವಯಸ್ಸು|বয়স)\D{0,8}(\d{1,2})",
        r"(\d{1,2})\D{0,10}(?:years?\s*old|yrs?|साल|वर्ष|उम्र|वर्षांचा|वर्षांची|వయస్సు|வயது|ವಯಸ್ಸು|বছর)",
    ]
    for pat in age_patterns:
        match = re.search(pat, text_lower)
        if match:
            age_val = int(match.group(1))
            if 1 <= age_val <= 110:
                detected_age = age_val
                break

    # 3. Concept matching (occupation/beneficiary + need)
    matched_concepts: list[str] = []
    matched_categories: list[str] = []
    matched_beneficiaries: list[str] = []
    occupations: list[str] = []
    needs: list[str] = []

    tokens = re.findall(r"[\w\u0900-\u0DFF]+", text_lower)
    for concept, meta in CONCEPT_KEYWORDS.items():
        hit = any(term in text_lower for term in meta["terms"])
        if not hit:
            # Fuzzy fallback for slight transliteration/spelling typos
            for tok in tokens:
                if len(tok) >= 4 and difflib.get_close_matches(tok, meta["terms"], n=1, cutoff=0.84):
                    hit = True
                    break
        if hit:
            matched_concepts.append(concept)
            matched_categories.extend(meta["categories"])
            matched_beneficiaries.extend(meta["beneficiaries"])
            if "occupation" in meta:
                occupations.append(meta["occupation"])
            if "need" in meta:
                needs.append(meta["need"])

    # Age-driven hints if occupation/need wasn't explicit
    if detected_age is not None:
        if detected_age >= 60 and "pension_elderly" not in matched_concepts:
            matched_concepts.append("pension_elderly")
            occupations.append("Senior Citizen")
            needs.append("Pension / Social Security")
        elif 13 <= detected_age <= 24 and not occupations:
            occupations.append("Youth / Student")

    occupation_str = ", ".join(dict.fromkeys(occupations)) if occupations else "General Citizen"
    need_str = ", ".join(dict.fromkeys(needs)) if needs else _infer_need_fallback(matched_concepts, query)

    return {
        "language": detected_lang,
        "state": detected_state or "All India / Not specified",
        "age": detected_age,
        "occupation_or_beneficiary": occupation_str,
        "need": need_str,
        "matched_concepts": matched_concepts,
        "categories": list(dict.fromkeys(matched_categories)),
        "beneficiaries": list(dict.fromkeys(matched_beneficiaries)),
    }


def _infer_need_fallback(concepts: list[str], query: str) -> str:
    if "farmer" in concepts:
        return "Farmer Financial Support / Crop Assistance"
    if "student" in concepts:
        return "Education / Scholarship / Skill Support"
    if "women" in concepts:
        return "Women & Girl Child Welfare"
    return (query or "").strip()[:80] or "Government Welfare Scheme"


def _strip_json_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _call_llm_json(prompt: str, provider: str) -> dict[str, Any] | None:
    """Invoke Gemini or OpenAI and parse JSON response."""
    gemini_key, openai_key = _get_api_keys()

    if provider == "gemini" and gemini_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key)
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            text = _strip_json_fences(response.text or "")
            return json.loads(text)
        except Exception:
            return None

    if provider == "openai" and openai_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_key)
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(_strip_json_fences(content))
        except Exception:
            return None

    return None


def is_safe_select_sql(sql: str, allowed_tables: set[str]) -> bool:
    """Verify SQL is a single read-only SELECT statement against known tables."""
    if not sql or not isinstance(sql, str):
        return False
    stripped = sql.strip().rstrip(";").strip()
    if ";" in stripped:
        return False
    upper = stripped.upper()
    if not (upper.startswith("SELECT ") or upper.startswith("WITH ")):
        return False
    forbidden = (
        "INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "CREATE ",
        "ATTACH ", "DETACH ", "REPLACE ", "PRAGMA ", "VACUUM", "REINDEX",
    )
    if any(tok in upper for tok in forbidden):
        return False
    if allowed_tables:
        referenced = re.findall(r"(?:FROM|JOIN)\s+[`\"\[]?([A-Za-z_][A-Za-z0-9_]*)[`\"\]]?", stripped, re.IGNORECASE)
        if not referenced or any(tbl not in allowed_tables for tbl in referenced):
            return False
    return True


def execute_safe_select(
    db_path: str,
    sql: str,
    params: list[Any] | tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    """Execute a validated read-only SQLite SELECT query."""
    schema_info = inspect_database_schema(db_path)
    allowed_tables = set(schema_info["tables"].keys())
    if not is_safe_select_sql(sql, allowed_tables):
        raise ValueError(f"Unsafe or invalid SQL query rejected: {sql}")

    conn = _connect_readonly(db_path)
    try:
        cur = conn.cursor()
        rows = cur.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _build_llm_sql_and_entities(
    query: str,
    schema_info: dict[str, Any],
    preferred_language: str | None,
    provider: str,
) -> dict[str, Any] | None:
    """Use LLM to extract entities and generate a read-only SQLite query."""
    schema_text = format_schema_for_prompt(schema_info)
    primary_table = schema_info.get("primary_table") or "schemes"
    lang_hint = preferred_language if preferred_language and preferred_language != "Auto-detect" else "Auto-detect from input"
    prompt = f"""You are a SQLite query planner for an Indian government schemes assistant.

Database Schema (dynamically inspected):
{schema_text}

User Query: {query!r}
Language Hint: {lang_hint}

Tasks:
1. Identify the exact input language among: Hindi, Marathi, Telugu, Tamil, Kannada, Bengali, Hinglish, English.
2. Extract:
   - state (English state name if mentioned or implied, else null)
   - age (integer if mentioned, else null)
   - occupation_or_beneficiary (short English phrase, e.g. "Farmer", "Student", "Women", "Street Vendor", "Senior Citizen")
   - need (short English phrase, e.g. "crop financial aid", "scholarship", "health insurance", "housing")
   - english_keywords (list of 4-8 English keywords/synonyms matching categories, beneficiaries, or scheme names in the table)
3. Write ONE safe read-only SQLite SELECT query on `{primary_table}` using ONLY columns from the schema above.
   - Select distinct or grouped scheme candidates (include scheme_name, category, state_or_central, target_beneficiary, official_website if those columns exist).
   - Use broad `LIKE '%keyword%'` conditions joined with `OR` so relevant rows aren't missed.
   - Do NOT use parameters (`?`); inline clean string literals.
   - Limit to 30 rows.

Return strict JSON with keys:
{{
  "language": "...",
  "state": "... or null",
  "age": null,
  "occupation_or_beneficiary": "...",
  "need": "...",
  "english_keywords": ["..."],
  "sql": "SELECT ..."
}}"""
    return _call_llm_json(prompt, provider)


def _normalize_scheme_key(name: str, website: str) -> str:
    """Deduplicate variant names pointing to the same official scheme."""
    site = (website or "").strip().lower().rstrip("/")
    site = re.sub(r"^https?://(www\.)?", "", site)
    cleaned = re.sub(r"\([^)]*\)", "", (name or "").lower())
    cleaned = re.sub(r"\b(programme|program|scheme|initiative|implementation|mission|yojana|abhiyan)\b", "", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", "", cleaned)
    if site and not site.endswith("/nep"):
        return site
    return cleaned or site


def _fallback_sql_and_rank(
    db_path: str,
    schema_info: dict[str, Any],
    entities: dict[str, Any],
    query: str,
    top_k: int = 3,
) -> tuple[str, list[dict[str, Any]]]:
    """Build a dynamic SQLite SELECT query and rank top-k distinct schemes."""
    table = schema_info["primary_table"]
    roles = schema_info["column_roles"]
    name_col = roles.get("scheme_name")
    cat_col = roles.get("category")
    ben_col = roles.get("target_beneficiary")
    level_col = roles.get("state_or_central")
    web_col = roles.get("official_website")
    q_en_col = roles.get("question_english")
    a_col = roles.get("answer")
    a_en_col = roles.get("answer_english")

    search_terms: list[str] = []
    if entities.get("state") and entities["state"] != "All India / Not specified":
        search_terms.append(entities["state"])
        if entities["state"] == "Madhya Pradesh":
            search_terms.append("MP")
        if entities["state"] == "Andhra Pradesh":
            search_terms.append("Telangana")
    for c in entities.get("categories", []):
        search_terms.append(c)
    for b in entities.get("beneficiaries", []):
        search_terms.append(b)
    for tok in re.findall(r"[A-Za-z]{3,}|[\u0900-\u0DFF]{2,}", query or ""):
        search_terms.append(tok)
    search_terms = list(dict.fromkeys(t for t in search_terms if t))[:12]

    # Build a clean, dynamically-columned SQL SELECT query
    select_cols = [c for c in [name_col, cat_col, level_col, ben_col, web_col, q_en_col, a_col, a_en_col] if c]
    select_clause = ", ".join(f'"{c}"' for c in dict.fromkeys(select_cols))

    searchable_cols = [c for c in [name_col, cat_col, ben_col, q_en_col, a_col, a_en_col] if c]
    where_parts: list[str] = []
    params: list[str] = []
    for term in search_terms[:8]:
        safe_like = f"%{term}%"
        or_cols = " OR ".join(f'LOWER(COALESCE("{c}", "")) LIKE LOWER(?)' for c in searchable_cols)
        where_parts.append(f"({or_cols})")
        params.extend([safe_like] * len(searchable_cols))

    where_sql = f"WHERE {' OR '.join(where_parts)}" if where_parts else ""
    sql = f'SELECT {select_clause} FROM "{table}" {where_sql} LIMIT 2500'

    rows = execute_safe_select(db_path, sql, params)
    if not rows:
        # Broad fallback across all schemes if narrow filter matched 0
        fallback_sql = f'SELECT {select_clause} FROM "{table}" LIMIT 3500'
        rows = execute_safe_select(db_path, fallback_sql)
        sql = fallback_sql

    ranked = _score_and_deduplicate_rows(rows, roles, entities, query, top_k=top_k)
    return sql, ranked


def _score_and_deduplicate_rows(
    rows: list[dict[str, Any]],
    roles: dict[str, str],
    entities: dict[str, Any],
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Score candidate rows by state, beneficiary, category, and need, returning top k distinct schemes."""
    name_col = roles.get("scheme_name", "scheme_name")
    cat_col = roles.get("category", "category")
    ben_col = roles.get("target_beneficiary", "target_beneficiary")
    level_col = roles.get("state_or_central", "state_or_central")
    web_col = roles.get("official_website", "official_website")
    q_en_col = roles.get("question_english", "question_english")
    a_col = roles.get("answer", "answer")
    a_en_col = roles.get("answer_english", "answer_english")

    state = entities.get("state")
    has_state = bool(state and state != "All India / Not specified")
    concepts = entities.get("matched_concepts", [])
    cat_targets = [c.lower() for c in entities.get("categories", [])]
    ben_targets = [b.lower() for b in entities.get("beneficiaries", [])]
    query_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]{3,}", query or "")]

    # Group rows by canonical scheme key
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        sname = str(row.get(name_col) or "").strip()
        if not sname:
            continue
        website = str(row.get(web_col) or "").strip()
        key = _normalize_scheme_key(sname, website)
        entry = grouped.setdefault(
            key,
            {
                "scheme_name": sname,
                "category": str(row.get(cat_col) or ""),
                "state_or_central": str(row.get(level_col) or ""),
                "target_beneficiary": str(row.get(ben_col) or ""),
                "official_website": website,
                "score": 0.0,
                "elig_snippets": [],
                "doc_snippets": [],
                "general_snippets": [],
            },
        )
        # Prefer fuller canonical scheme name
        if len(sname) > len(entry["scheme_name"]) and "(" in sname:
            entry["scheme_name"] = sname

        q_en = str(row.get(q_en_col) or "")
        ans_hi = str(row.get(a_col) or "")
        ans_en = str(row.get(a_en_col) or "")
        q_lower = q_en.lower()
        if "eligib" in q_lower and ans_hi:
            entry["elig_snippets"].append((q_en, ans_hi, ans_en))
        elif "document" in q_lower and ans_hi:
            entry["doc_snippets"].append((q_en, ans_hi, ans_en))
        elif ans_hi:
            entry["general_snippets"].append((q_en, ans_hi, ans_en))

    # Score each distinct scheme
    for entry in grouped.values():
        sname_l = entry["scheme_name"].lower()
        cat_l = entry["category"].lower()
        ben_l = entry["target_beneficiary"].lower()
        level_l = entry["state_or_central"].lower()
        combined_text = " ".join(
            [sname_l, cat_l, ben_l]
            + [s[0].lower() for s in (entry["elig_snippets"] + entry["general_snippets"])[:4]]
        )

        score = 0.0
        # State match boost
        if has_state:
            state_l = state.lower()
            state_aliases = [state_l]
            if state == "Madhya Pradesh":
                state_aliases.append("(mp)")
            if state_l in sname_l or any(al in sname_l for al in state_aliases):
                score += 18.0
            elif level_l == "state":
                # Penalize mismatched state-only schemes when user specified a different state
                score -= 10.0
            elif level_l == "central":
                score += 2.0

        # Category & beneficiary alignment
        for idx, cat in enumerate(cat_targets):
            if cat == cat_l or cat in cat_l:
                score += max(6.0 - idx * 1.2, 2.5)
        for ben in ben_targets:
            if ben in ben_l or any(w in ben_l for w in ben.split() if len(w) > 3):
                score += 5.5

        # Concept-specific keyword boost on scheme name & text
        for concept in concepts:
            for term in CONCEPT_KEYWORDS[concept]["terms"]:
                if term.isascii() and len(term) >= 4 and term in sname_l:
                    score += 7.0
                elif term in combined_text:
                    score += 1.5

        # Age-aware relevance tuning (avoid child-only schemes for adults & vice versa)
        age_val = entities.get("age")
        if isinstance(age_val, int):
            if age_val >= 20 and any(cw in ben_l for cw in ("girl child", "school children", "primary school")):
                score -= 6.0
            if age_val >= 18 and "business_loan" in concepts and any(
                bw in combined_text for bw in ("mudra", "stand up", "loan", "entrepreneur", "vendor", "artisan", "startup")
            ):
                score += 6.0

        # Direct query token matches & fuzzy similarity
        for tok in query_tokens:
            if tok in sname_l:
                score += 6.0
            elif tok in combined_text:
                score += 2.0
            elif difflib.get_close_matches(tok, sname_l.split(), n=1, cutoff=0.82):
                score += 4.0

        # Slight bonus for having dedicated eligibility/doc info
        if entry["elig_snippets"]:
            score += 0.5
        if entry["doc_snippets"]:
            score += 0.5
        entry["score"] = score

    ranked = sorted(grouped.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def _enrich_scheme_rows_from_db(
    db_path: str,
    schema_info: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fetch dedicated eligibility and document Q&A rows for the top 3 schemes."""
    table = schema_info["primary_table"]
    roles = schema_info["column_roles"]
    name_col = roles.get("scheme_name", "scheme_name")
    web_col = roles.get("official_website", "official_website")
    q_en_col = roles.get("question_english", "question_english")
    a_col = roles.get("answer", "answer")
    a_en_col = roles.get("answer_english", "answer_english")
    ben_col = roles.get("target_beneficiary", "target_beneficiary")
    cat_col = roles.get("category", "category")
    level_col = roles.get("state_or_central", "state_or_central")

    conn = _connect_readonly(db_path)
    try:
        cur = conn.cursor()
        enriched: list[dict[str, Any]] = []
        for item in candidates:
            website = item.get("official_website", "")
            sname = item.get("scheme_name", "")
            if website and web_col:
                rows = cur.execute(
                    f'SELECT "{name_col}", "{cat_col}", "{level_col}", "{ben_col}", '
                    f'"{web_col}", "{q_en_col}", "{a_col}", "{a_en_col}" '
                    f'FROM "{table}" WHERE "{web_col}" = ? LIMIT 25',
                    (website,),
                ).fetchall()
            else:
                rows = cur.execute(
                    f'SELECT "{name_col}", "{cat_col}", "{level_col}", "{ben_col}", '
                    f'"{web_col}", "{q_en_col}", "{a_col}", "{a_en_col}" '
                    f'FROM "{table}" WHERE "{name_col}" = ? LIMIT 25',
                    (sname,),
                ).fetchall()

            elig_texts: list[str] = []
            doc_texts: list[str] = []
            general_texts: list[str] = []
            for r in rows:
                q_l = str(r[q_en_col] or "").lower()
                ans_hi = str(r[a_col] or "")
                ans_en = str(r[a_en_col] or "")
                if "eligib" in q_l and len(elig_texts) < 2:
                    elig_texts.append(ans_hi[:1100] + "\n" + ans_en[:400])
                elif "document" in q_l and len(doc_texts) < 2:
                    doc_texts.append(ans_hi[:1100] + "\n" + ans_en[:400])
                elif len(general_texts) < 2:
                    general_texts.append(ans_hi[:700])

            item_copy = dict(item)
            item_copy["raw_eligibility_context"] = "\n---\n".join(elig_texts or general_texts)
            item_copy["raw_documents_context"] = "\n---\n".join(doc_texts or general_texts)
            enriched.append(item_copy)
        return enriched
    finally:
        conn.close()


def _extract_documents_offline(raw_context: str, language: str) -> str:
    """Extract clean canonical document names from DB answer text and localize."""
    ctx_lower = (raw_context or "").lower()
    found: list[str] = []
    for canonical, patterns in DOCUMENT_PATTERNS:
        if any(p in ctx_lower for p in patterns):
            localized = DOCUMENT_TRANSLATIONS[canonical].get(language, canonical)
            found.append(localized)
        if len(found) >= 5:
            break
    if not found:
        defaults = ["Aadhaar Card", "Bank Account / Passbook", "Residence / Domicile Certificate", "Passport-size Photographs"]
        found = [DOCUMENT_TRANSLATIONS[d].get(language, d) for d in defaults]
    return ", ".join(found)


def _format_eligibility_offline(item: dict[str, Any], language: str) -> str:
    """Create a concise, accurate eligibility summary in the target language."""
    beneficiary = item.get("target_beneficiary") or "Eligible Indian Citizens"
    level = item.get("state_or_central") or "Central"
    category = item.get("category") or "Welfare"

    templates: dict[str, str] = {
        "English": f"{beneficiary} ({level} scheme, {category} sector); Indian citizen meeting scheme income/age criteria.",
        "Hindi": f"{beneficiary} ({level} योजना, {category}); निर्धारित आयु व आय मानदंड पूरा करने वाले पात्र नागरिक।",
        "Marathi": f"{beneficiary} ({level} योजना, {category}); विहित वय आणि उत्पन्न निकष पूर्ण करणारे पात्र नागरिक.",
        "Telugu": f"{beneficiary} ({level} పథకం, {category}); నిర్ణీత వయస్సు మరియు ఆదాయ అర్హతలు ఉన్న భారతీయ పౌరులు.",
        "Tamil": f"{beneficiary} ({level} திட்டம், {category}); நிர்ணயிக்கப்பட்ட வயது மற்றும் வருமானத் தகுதியுள்ள குடிமக்கள்.",
        "Kannada": f"{beneficiary} ({level} ಯೋಜನೆ, {category}); ನಿಗದಿತ ವಯಸ್ಸು ಮತ್ತು ಆದಾಯ ಮಾನದಂಡ ಪೂರೈಸುವ ನಾಗರಿಕರು.",
        "Bengali": f"{beneficiary} ({level} প্রকল্প, {category}); নির্ধারিত বয়স ও আয়ের যোগ্যতা পূরণকারী নাগরিক।",
        "Hinglish": f"{beneficiary} ({level} scheme, {category}); jo nirdharit age aur income criteria poore karte hain.",
    }
    return templates.get(language, templates["English"])


def _synthesize_with_llm(
    query: str,
    language: str,
    entities: dict[str, Any],
    enriched_schemes: list[dict[str, Any]],
    provider: str,
) -> list[dict[str, str]] | None:
    """Use LLM to produce strictly 4 fields per scheme in the user's exact language."""
    if provider == "fallback" or not enriched_schemes:
        return None

    context_blocks = []
    for idx, sc in enumerate(enriched_schemes, 1):
        context_blocks.append(
            f"Scheme {idx}:\n"
            f"- scheme_name: {sc['scheme_name']}\n"
            f"- state_or_central: {sc['state_or_central']}\n"
            f"- target_beneficiary: {sc['target_beneficiary']}\n"
            f"- official_website: {sc['official_website']}\n"
            f"- eligibility_context: {sc.get('raw_eligibility_context', '')[:1200]}\n"
            f"- documents_context: {sc.get('raw_documents_context', '')[:1200]}"
        )

    prompt = f"""You are YojanaSetu Voice, an Indian civic services assistant.

User spoken query: {query!r}
Target reply language: {language}
Extracted user profile: state={entities.get('state')}, age={entities.get('age')}, occupation={entities.get('occupation_or_beneficiary')}, need={entities.get('need')}

Below are the top {len(enriched_schemes)} matching schemes from SQLite.
For EACH scheme, return ONLY 4 fields in {language} (if Hinglish, write natural Romanized Hindi+English; if Hindi/Marathi/Telugu/Tamil/Kannada/Bengali, write in that native script):
1. "scheme_name": Official scheme name (keep official English/Hindi name clear).
2. "eligibility": 1-2 crisp sentences in {language} explaining who is eligible (mention age/income/state rules from context if present).
3. "documents_needed": Concise comma-separated list of required documents in {language} grounded in the context.
4. "official_website": Exact URL from the context (do NOT alter or invent URLs).

Return strict JSON:
{{
  "schemes": [
    {{
      "scheme_name": "...",
      "eligibility": "...",
      "documents_needed": "...",
      "official_website": "..."
    }}
  ]
}}

Database Context:
{chr(10).join(context_blocks)}"""

    result = _call_llm_json(prompt, provider)
    if not result or not isinstance(result.get("schemes"), list):
        return None

    cleaned: list[dict[str, str]] = []
    for idx, item in enumerate(result["schemes"][:3]):
        if not isinstance(item, dict):
            continue
        fallback_site = enriched_schemes[idx]["official_website"] if idx < len(enriched_schemes) else ""
        cleaned.append(
            {
                "scheme_name": str(item.get("scheme_name") or enriched_schemes[idx]["scheme_name"]).strip(),
                "eligibility": str(item.get("eligibility") or "").strip(),
                "documents_needed": str(item.get("documents_needed") or "").strip(),
                "official_website": str(item.get("official_website") or fallback_site).strip(),
            }
        )
    return cleaned if cleaned else None


def format_schemes_reply(schemes: list[dict[str, str]], language: str) -> tuple[str, str]:
    """Format strictly the 4 allowed fields in the detected language + TTS summary."""
    labels = FIELD_LABELS.get(language, FIELD_LABELS["English"])
    blocks: list[str] = []
    tts_parts: list[str] = []
    for idx, sc in enumerate(schemes, 1):
        blocks.append(
            f"{idx}. **{labels['scheme_name']}:** {sc['scheme_name']}\n"
            f"   - **{labels['eligibility']}:** {sc['eligibility']}\n"
            f"   - **{labels['documents_needed']}:** {sc['documents_needed']}\n"
            f"   - **{labels['official_website']}:** {sc['official_website']}"
        )
        tts_parts.append(
            f"{idx}. {sc['scheme_name']}. "
            f"{labels['eligibility']}: {sc['eligibility']}. "
            f"{labels['documents_needed']}: {sc['documents_needed']}."
        )
    return "\n\n".join(blocks), " ".join(tts_parts)


def query_schemes(
    query: str,
    db_path: str = "schemes.db",
    preferred_language: str | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Run the full NL-to-SQLite pipeline and return top 3 matching schemes."""
    clean_query = (query or "").strip()
    if not clean_query:
        raise ValueError("Query cannot be empty.")

    schema_info = inspect_database_schema(db_path)
    provider = get_active_provider()
    language = detect_language(clean_query, preferred_language)
    entities = extract_query_entities(clean_query, language)

    sql_used = ""
    ranked_candidates: list[dict[str, Any]] = []

    if provider in ("gemini", "openai"):
        llm_plan = _build_llm_sql_and_entities(clean_query, schema_info, preferred_language, provider)
        if llm_plan:
            if not preferred_language or preferred_language == "Auto-detect":
                cand_lang = llm_plan.get("language")
                if cand_lang in FIELD_LABELS:
                    language = cand_lang
                    entities["language"] = cand_lang
            if llm_plan.get("state"):
                entities["state"] = str(llm_plan["state"])
            if llm_plan.get("age") is not None:
                entities["age"] = llm_plan["age"]
            if llm_plan.get("occupation_or_beneficiary"):
                entities["occupation_or_beneficiary"] = str(llm_plan["occupation_or_beneficiary"])
            if llm_plan.get("need"):
                entities["need"] = str(llm_plan["need"])

            llm_sql = str(llm_plan.get("sql") or "").strip()
            allowed_tables = set(schema_info["tables"].keys())
            if llm_sql and is_safe_select_sql(llm_sql, allowed_tables):
                try:
                    llm_rows = execute_safe_select(db_path, llm_sql)
                    if llm_rows:
                        sql_used = llm_sql
                        ranked_candidates = _score_and_deduplicate_rows(
                            llm_rows, schema_info["column_roles"], entities, clean_query, top_k=top_k
                        )
                except Exception:
                    ranked_candidates = []

    if len(ranked_candidates) < top_k:
        fallback_sql, fallback_ranked = _fallback_sql_and_rank(
            db_path, schema_info, entities, clean_query, top_k=top_k
        )
        if not ranked_candidates:
            sql_used = fallback_sql
            ranked_candidates = fallback_ranked
        else:
            seen_keys = {_normalize_scheme_key(r["scheme_name"], r["official_website"]) for r in ranked_candidates}
            for item in fallback_ranked:
                k = _normalize_scheme_key(item["scheme_name"], item["official_website"])
                if k not in seen_keys:
                    ranked_candidates.append(item)
                    seen_keys.add(k)
                if len(ranked_candidates) >= top_k:
                    break

    enriched = _enrich_scheme_rows_from_db(db_path, schema_info, ranked_candidates[:top_k])

    synthesized = _synthesize_with_llm(clean_query, language, entities, enriched, provider)
    if synthesized:
        final_schemes = synthesized
    else:
        final_schemes = [
            {
                "scheme_name": item["scheme_name"],
                "eligibility": _format_eligibility_offline(item, language),
                "documents_needed": _extract_documents_offline(item.get("raw_documents_context", ""), language),
                "official_website": item["official_website"],
            }
            for item in enriched
        ]

    formatted_reply, tts_text = format_schemes_reply(final_schemes, language)
    labels = FIELD_LABELS.get(language, FIELD_LABELS["English"])

    return {
        "query": clean_query,
        "language": language,
        "provider": provider,
        "entities": {
            "state": entities.get("state"),
            "age": entities.get("age"),
            "occupation_or_beneficiary": entities.get("occupation_or_beneficiary"),
            "need": entities.get("need"),
        },
        "sql_query": sql_used,
        "labels": labels,
        "schemes": final_schemes,
        "formatted_reply": formatted_reply,
        "tts_text": tts_text,
    }


def transcribe_audio(
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
    preferred_language: str | None = None,
) -> dict[str, Any]:
    """Transcribe spoken audio bytes using Gemini or OpenAI when configured."""
    if not audio_bytes:
        return {"text": "", "language": preferred_language or "Auto-detect", "error": "Empty audio recording."}

    provider = get_active_provider()
    gemini_key, openai_key = _get_api_keys()
    lang_note = (
        f"The speaker is likely speaking in {preferred_language}."
        if preferred_language and preferred_language != "Auto-detect"
        else "Detect the Indian language (Hindi, Marathi, Telugu, Tamil, Kannada, Bengali, Hinglish, or English)."
    )

    if provider == "gemini" and gemini_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key)
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            prompt = (
                f"Transcribe this spoken user query accurately in its original script/language. "
                f"{lang_note} Return strict JSON: {{\"text\": \"...\", \"language\": \"...\"}}."
            )
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            parsed = json.loads(_strip_json_fences(response.text or "{}"))
            return {
                "text": str(parsed.get("text") or "").strip(),
                "language": str(parsed.get("language") or preferred_language or "Auto-detect"),
                "error": None,
            }
        except Exception as exc:
            return {"text": "", "language": preferred_language or "Auto-detect", "error": f"Gemini transcription failed: {exc}"}

    if provider == "openai" and openai_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_key)
            ext = ".wav" if "wav" in mime_type else ".mp3"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                with open(tmp_path, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                    )
                text = (transcript.text or "").strip()
                return {
                    "text": text,
                    "language": detect_language(text, preferred_language),
                    "error": None,
                }
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except Exception as exc:
            return {"text": "", "language": preferred_language or "Auto-detect", "error": f"OpenAI Whisper failed: {exc}"}

    return {
        "text": "",
        "language": preferred_language or "Auto-detect",
        "error": "No GEMINI_API_KEY or OPENAI_API_KEY configured in .env for raw audio STT. Dictate directly with Wispr Flow into the text box or add an API key.",
    }


def synthesize_speech(text: str, language: str = "English") -> tuple[bytes | None, str | None]:
    """Convert reply text to MP3 bytes via gTTS for hands-free listening."""
    clean = (text or "").strip()
    if not clean:
        return None, "Nothing to read aloud."

    lang_code = TTS_LANGUAGE_CODES.get(language, "en")
    try:
        from gtts import gTTS

        buf = io.BytesIO()
        tts = gTTS(text=clean, lang=lang_code, slow=False)
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read(), None
    except Exception as exc:
        return None, f"Audio playback unavailable (check internet or gTTS install): {exc}"
