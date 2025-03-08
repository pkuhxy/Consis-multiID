import base64
from zhipuai import ZhipuAI

video_path = "/storage/hxy/ID/data/data_processor/test/step2_test_videos/gm472964359-19589472_part2.mp4"
with open(video_path, 'rb') as video_file:

    # import ipdb;ipdb.set_trace()
    video_base = base64.b64encode(video_file.read()).decode('utf-8')

temp = (
  "Please generate a comprehensive caption for the following video, describing various aspects, including but not limited to: "
  "1. The main theme and setting of the image (such as location, time of day, weather conditions, etc.) "
  "2. Key objects and their characteristics (such as color, shape, size, etc.) "
  "3. Relationships and interactions between objects (such as positioning, actions, etc.) "
  "4. Any people present and their emotions or activities (such as expressions, postures, etc.) "
  "5. Background and environmental details (such as architecture, natural scenery, etc.) "
  "6. Motion of the Subject: The movement of people or objects in the video. Use verbs that describe movement. "
  "7. Camera motion control: zoom in, zoom out, push in, pull out, pan right, pan left, truck right, truck left, tilt up, tilt down, pedestal up, pedestal down, arc shot,  tracking shot, static shot, and handheld shot. "
  "Do not describe imagined content. Only describe what can be determined from the video. Avoid listing things. Do not use abstract concepts (love, hate, justice, infinity, joy) as subjects. Use concrete nouns (human, cup, dog, planet, headphones) for more accurate results. Use verbs to describe the movement and changes of the subject or people. Write your prompts in plain, conversational language. Start your description directly with the main subject, typically a noun. Without \"\n\", subheading and title. "
  "Please describe the content of the video and the changes that occur, in chronological order:"
)

client = ZhipuAI(api_key="388291dd58fd4e1ab75b43664825a712.PzOLjKEDTQrKWDHP") # 填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-4v-plus-0111",  # 填写需要调用的模型名称
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "video_url",
            "video_url": {
                "url" : video_base
            }
          },
          {
            "type": "text",
            "text": temp
          }
        ]
      }
    ]
)
print(response.choices[0].message)