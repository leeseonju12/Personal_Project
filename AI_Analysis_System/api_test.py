import os
from google import genai

# API 키 설정
client = genai.Client(api_key="")

# 응답 테스트
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="안녕! 간단히 자기소개 해줘."
)

print("✅ API 키 정상 작동!")
print(f"응답: {response.text}")