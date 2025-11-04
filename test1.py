from zai import ZhipuAiClient

client = ZhipuAiClient(api_key="409c732b24c344eb9525919467821b13.Ep4NKHIocKvELO48")  # 请填写您自己的 API Key
prompt='以色列为什么喜欢战争'
response = client.chat.completions.create(
    model="glm-4.5",
    messages=[
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "我是人工智能助手"},
        {"role": "user", "content": prompt}
    ],
    stream=True,
    thinking={
        "type": "enabled",    # 启用深度思考模式
    },
    max_tokens=4096,          # 最大输出tokens
    temperature=0.6           # 控制输出的随机性


)
for chunk in response:
    print(chunk.choices[0].delta.content, end='')
