from langchain.llms.base import LLM
from zhipuai import ZhipuAI
from langchain_core.messages.ai import AIMessage
from typing import Optional, List

# 替换为你的智谱AI API Key
zhipuai_api_key = "409c732b24c344eb9525919467821b13.Ep4NKHIocKvELO48"


class ChatGLM4_5(LLM):
    history: List[dict] = []
    client: object = None

    def __init__(self):
        super().__init__()
        # 初始化ZhipuAI客户端
        self.client = ZhipuAI(api_key=zhipuai_api_key)
        print("ChatGLM4_5 initialized with ZhipuAI client.")

    @property
    def _llm_type(self):
        return "ChatGLM4_5"

    def invoke(self, prompt: str, history: Optional[List[dict]] = None) -> AIMessage:
        if history is None:
            history = []
        history.append({"role": "user", "content": prompt})

        print(f"Sending prompt: {prompt}")  # 打印发送的prompt
        print(f"History: {history}")  # 打印发送的history

        try:
            # 调用智谱AI的API接口
            response = self.client.chat.completions.create(
                model="glm-4.5",  # 替换为ChatGLM4.5对应的模型ID，需查阅智谱AI文档确认
                messages=history
            )

            result = response.choices[0].message.content
            print(f"Response: {result}")  # 打印AI返回的结果
            return AIMessage(content=result)
        except Exception as e:
            print(f"Error occurred: {e}")
            raise e

    def _call(self, prompt: str, history: Optional[List[dict]] = None) -> AIMessage:
        # 实现 _call 方法，调用 invoke
        return self.invoke(prompt, history)

    def __call__(self, prompt: str, history: Optional[List[dict]] = None) -> AIMessage:
        return self.invoke(prompt, history)

    def stream(self, prompt: str, history: Optional[List[dict]] = None):
        if history is None:
            history = []
        history.append({"role": "user", "content": prompt})

        print(f"Streaming prompt: {prompt}")  # 打印正在流式处理的prompt
        print(f"History: {history}")  # 打印正在流式处理的history

        try:
            # 启用流式响应
            response = self.client.chat.completions.create(
                model="glm-4.5",  # 替换为ChatGLM4.5对应的模型ID
                messages=history,
                stream=True  # 启用流式响应
            )

            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Error occurred during streaming: {e}")
            raise e


# 测试代码
if __name__ == "__main__":
    # 初始化ChatGLM4_5模型
    llm = ChatGLM4_5()

    # 发送简单的消息进行测试
    prompt = "你好，ChatGLM4.5！"
    response = llm.invoke(prompt)
    print(f"AI Response: {response.content}")  # 打印AI返回的内容

    # 流式响应测试（如果想测试流式输出）
    print("Testing streaming response:")
    for chunk in llm.stream(prompt):
        print(chunk)
