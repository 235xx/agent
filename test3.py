from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage
from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation
from typing import Optional, List, Dict, Any
import requests
import json
from pydantic import Field


# 自定义ChatGLM LLM类
class ChatGLM(LLM):
    api_url: str = Field(...)
    api_key: str = Field(...)

    def __init__(self, api_url: str, api_key: str, **kwargs):
        super().__init__(
            api_url=api_url,
            api_key=api_key,
            **kwargs
        )

    @property
    def _llm_type(self) -> str:
        return "chatglm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 根据智谱AI API格式调整请求数据
        data = {
            "model": "glm-4.5",  # 使用正确的模型名称
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }

        if stop:
            data["stop"] = stop

        response = requests.post(self.api_url, headers=headers, json=data)

        if response.status_code != 200:
            raise Exception(f"API请求失败: {response.status_code} - {response.text}")

        result = response.json()
        # 根据智谱AI API返回格式解析
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        return result.get("response", "")

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)


# 初始化ChatGLM
glm = ChatGLM(
    api_url="https://open.bigmodel.cn/api/paas/v4/chat/completions",  # 智谱AI官方API地址
    api_key="409c732b24c344eb9525919467821b13.Ep4NKHIocKvELO48"  # 您提供的API密钥
)


# 示例1: 基础提示词模板
def basic_prompt_example():
    template = "我想学习{topic}，请给我推荐3本入门书籍"
    prompt = PromptTemplate(
        input_variables=["topic"],
        template=template
    )

    chain = LLMChain(llm=glm, prompt=prompt)
    result = chain.run(topic="机器学习")
    print("基础提示词示例结果:")
    print(result)
    print("\n" + "=" * 50 + "\n")


# 示例2: 聊天提示词模板
def chat_prompt_example():
    system_template = "你是一个专业的{role}，请用{style}的风格回答问题"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # 转换为字符串格式
    prompt_str = chat_prompt.format_prompt(
        role="Python开发工程师",
        style="简洁专业",
        text="解释一下什么是装饰器"
    ).to_string()

    response = glm(prompt_str)
    print("聊天提示词示例结果:")
    print(response)
    print("\n" + "=" * 50 + "\n")


# 示例3: 少样本学习
def few_shot_example():
    from langchain.prompts import FewShotPromptTemplate

    examples = [
        {"question": "2+2等于几？", "answer": "4"},
        {"question": "5-3等于几？", "answer": "2"}
    ]

    example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="问题: {question}\n答案: {answer}"
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="问题: {input}\n答案:",
        input_variables=["input"]
    )

    chain = LLMChain(llm=glm, prompt=few_shot_prompt)
    result = chain.run(input="3+4等于几？")
    print("少样本学习示例结果:")
    print(result)
    print("\n" + "=" * 50 + "\n")


# 示例4: 输出解析
def output_parsing_example():
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field

    class BookRecommendation(BaseModel):
        title: str = Field(description="书名")
        author: str = Field(description="作者")
        reason: str = Field(description="推荐理由")

    parser = PydanticOutputParser(pydantic_object=BookRecommendation)

    template = """推荐一本关于{topic}的书籍。
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["topic"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = LLMChain(llm=glm, prompt=prompt)
    result = chain.run(topic="人工智能")

    try:
        parsed_result = parser.parse(result)
        print("输出解析示例结果:")
        print(f"书名: {parsed_result.title}")
        print(f"作者: {parsed_result.author}")
        print(f"推荐理由: {parsed_result.reason}")
    except Exception as e:
        print(f"解析失败: {e}")
        print(f"原始输出: {result}")

    print("\n" + "=" * 50 + "\n")


# 运行所有示例
if __name__ == "__main__":
    print("LangChain与ChatGLM集成示例\n")

    # 运行示例
    basic_prompt_example()
    chat_prompt_example()
    few_shot_example()
    output_parsing_example()
