from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import os
import warnings
import sys
from typing import Optional, List, Dict, Any
import requests
import json
from pydantic import Field

from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage, LLMResult, Generation
from langchain.llms.base import LLM
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# 忽略LangChain的弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")


# ---------------------- 1. 自定义ChatGLM LLM类（无修改） ----------------------
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

        data = {
            "model": "glm-4.5",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000
        }

        if stop:
            data["stop"] = stop

        response = requests.post(self.api_url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"API请求失败: {response.status_code} - {response.text}")

        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        return result.get("response", "")

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)


# 初始化ChatGLM（需确认api_url和api_key有效性）
glm = ChatGLM(
    api_url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
    api_key="409c732b24c344eb9525919467821b13.Ep4NKHIocKvELO48"
)


# ---------------------- 2. 整合测试成功逻辑的自习室预定自动化类 ----------------------
class StudyRoomBookingTester:
    def __init__(self):
        self.driver = None
        # 可配置参数：完全沿用你测试成功的配置（关键！）
        self.username = "u3665742"  # 你的用户名
        self.password = "Zjm20020808"  # 你的密码
        # 元素定位器：全部替换为你测试成功的XPath/逻辑
        self.login_btn_xpath = "//input[@type='submit' or @type='button' or contains(@value, 'Login') or contains(@value, '登录')]"
        self.library_select_id = "main_ddlLibrary"
        self.library_option_xpath = "/html/body/form/div[5]/div/div[1]/table/tbody/tr[1]/td[2]/select/option[6]"  # Main Library
        self.facility_select_id = "main_ddlType"
        self.facility_type_xpath = "/html/body/form/div[5]/div/div[1]/table/tbody/tr[2]/td[2]/select/option[4]"  # Computer
        self.date_select_id = "main_ddlView"
        self.date_option_xpath = "/html/body/form/div[5]/div/div[1]/table/tbody/tr[3]/td[2]/select/option[3]"  # 10月1日
        self.query_btn_id = "main_btnGetResult"
        self.seat_xpath = "/html/body/form/div[5]/div/div[1]/div[4]/div/table/tbody/tr[2]/td[3]"  # 目标座位
        self.submit_btn_id = "main_btnSubmit"
        self.confirm_btn_id = "main_btnSubmitYes"

    def run_booking_test(self):
        """完整执行自习室预定自动化流程（整合你测试成功的逻辑）"""
        try:
            print("=== 开始执行自习室预定自动化测试 ===")

            # 1. 初始化浏览器（沿用你测试成功的Firefox配置）
            self.driver = webdriver.Firefox()
            self.driver.maximize_window()
            print("1. 浏览器初始化完成")

            # 2. 访问预定系统
            self.driver.get("https://booking.lib.hku.hk/Secure/FacilityStatusDate.aspx")
            print("2. 已打开图书馆预定系统页面")
            sleep(2)  # 沿用你测试成功的等待时间

            # 3. 登录：输入用户名（保留显式等待，提升稳定性）
            print("3. 开始登录 - 输入用户名")
            username_field = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.NAME, "userid"))
            )
            username_field.clear()
            username_field.send_keys(self.username)
            sleep(2)  # 沿用你测试成功的等待时间

            # 4. 登录：输入密码
            print("4. 登录 - 输入密码")
            password_field = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.NAME, "password"))
            )
            password_field.clear()
            password_field.send_keys(self.password)
            sleep(2)  # 沿用你测试成功的等待时间

            # 5. 登录：点击登录按钮（关键修改：用你测试成功的XPath）
            print("5. 登录 - 点击登录按钮")
            login_button = WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable((By.XPATH, self.login_btn_xpath))
            )
            login_button.click()
            sleep(5)  # 沿用你测试成功的登录等待时间
            print("6. 登录成功，进入预定页面")

            # 7. 选择图书馆（Main Library）- 沿用你测试成功的点击逻辑
            print("7. 选择图书馆 - Main Library")
            self.driver.find_element(By.ID, self.library_select_id).click()
            sleep(2)
            main_library_option = self.driver.find_element(By.XPATH, self.library_option_xpath)
            main_library_option.click()
            print("   ✅ 已选择Main Library")
            sleep(2)

            # 8. 选择设施类型（Computer）- 沿用你测试成功的逻辑
            print("8. 选择设施类型 - Computer")
            self.driver.find_element(By.ID, self.facility_select_id).click()
            sleep(2)
            self.driver.find_element(By.XPATH, self.facility_type_xpath).click()
            print("   ✅ 已选择Computer设施")
            sleep(2)

            # 9. 选择日期（10月1日）- 沿用你测试成功的逻辑
            print("9. 选择预定日期 - 10月1日")
            self.driver.find_element(By.ID, self.date_select_id).click()
            sleep(2)
            self.driver.find_element(By.XPATH, self.date_option_xpath).click()
            print("   ✅ 已选择10月1日")
            sleep(2)

            # 10. 点击查询按钮 - 沿用你测试成功的ID定位
            print("10. 点击查询按钮，加载可用座位")
            query_button = self.driver.find_element(By.ID, self.query_btn_id)
            query_button.click()
            sleep(6)  # 沿用你测试成功的加载等待时间
            print("   ✅ 座位数据加载完成")

            # 11. 选择目标座位 - 沿用你测试成功的XPath
            print("11. 选择目标座位")
            target_seat = self.driver.find_element(By.XPATH, self.seat_xpath)
            target_seat.click()
            print("   ✅ 已选择目标座位")
            sleep(2)

            # 12. 提交预约
            print("12. 提交预约请求")
            submit_btn = self.driver.find_element(By.ID, self.submit_btn_id)
            submit_btn.click()
            sleep(2)
            print("   ✅ 预约请求已提交")

            # 13. 确认预约（补充点击，确保流程闭环）
            print("13. 确认预约")
            confirm_btn = self.driver.find_element(By.ID, self.confirm_btn_id)
            confirm_btn.click()
            sleep(3)
            print("   ✅ 预约确认完成")

            # 14. 流程结束
            success_msg = "=== 自习室预定自动化测试执行成功！已完成所有步骤 ==="
            print(success_msg)
            return success_msg

        except Exception as e:
            error_msg = f"=== 测试执行失败：{str(e)} ==="
            print(error_msg)
            return error_msg

        finally:
            # 无论成功/失败，都关闭浏览器（释放资源）
            if self.driver:
                print("=== 关闭浏览器，释放资源 ===")
                self.driver.quit()


# ---------------------- 3. 工具与Agent初始化（无修改，确保调用正确） ----------------------
# 创建测试工具实例
tester = StudyRoomBookingTester()


# 定义工具函数（调用整合后的run_booking_test方法）
def run_booking_tests(query):
    """运行完整的自习室预定自动化测试"""
    return tester.run_booking_test()


def check_booking_status(query):
    """检查预定系统状态（保持原有逻辑）"""
    try:
        driver = webdriver.Firefox()
        driver.get("https://booking.lib.hku.hk/Secure/FacilityStatusDate.aspx")
        sleep(3)
        if "Facility Status" in driver.title:
            status_msg = "✅ 图书馆预定系统当前可正常访问"
        else:
            status_msg = "⚠️ 图书馆预定系统页面标题异常，可能存在问题"
        driver.quit()
        return status_msg
    except Exception as e:
        return f"❌ 检查预定系统状态失败：{str(e)}"


def get_booking_help(query):
    """获取预定帮助信息（保持原有逻辑）"""
    return """
    🏫 自习室预定系统使用帮助：
    1. 手动预定：访问系统 → 登录 → 选图书馆（Main Library）→ 选设施（Computer）→ 选日期 → 选座位 → 提交确认
    2. 自动化测试：发送"测试预定系统"或"帮我预定自习室"，将自动执行完整预定流程
    3. 状态检查：发送"检查系统状态"，可查询预定系统是否正常

    ⚠️ 注意：自动化测试需确保Firefox浏览器和geckodriver已正确安装，且用户名/密码有效。
    """


# 创建工具列表（保持原有结构）
tools = [
    Tool(
        name="RunBookingTests",
        func=run_booking_tests,
        description="当用户需要执行自习室预定自动化测试时使用，如用户说'测试预定系统'、'帮我预定自习室'"
    ),
    Tool(
        name="CheckBookingStatus",
        func=check_booking_status,
        description="当用户询问预定系统是否可用时使用，如用户说'系统能正常用吗'、'检查系统状态'"
    ),
    Tool(
        name="GetBookingHelp",
        func=get_booking_help,
        description="当用户需要预定流程指导时使用，如用户说'怎么预定自习室'、'需要预定帮助'"
    )
]

# 初始化Agent（保持原有系统提示词和配置）
system_prompt = """你是自习室预定系统专属助手，核心功能是自动化测试和预定指导。
1. 自动化测试：用户说"测试预定"、"自动预定"、"帮我订自习室"时，必须调用RunBookingTests工具，执行完整预定流程
2. 系统检查：用户问"系统好着吗"、"能登录吗"时，调用CheckBookingStatus工具
3. 帮助指导：用户问"怎么订"、"步骤是什么"时，调用GetBookingHelp工具
4. 结果反馈：执行工具后，用简洁语言告知用户结果（成功/失败原因），避免技术术语过多。
"""

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(
    tools,
    glm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    agent_kwargs={"system_message": system_prompt}
)


# ---------------------- 4. Flask Web服务（无修改，确保前端正常调用） ----------------------
app = Flask(__name__)
CORS(app)  # 允许跨域请求


@app.route('/')
def index():
    """提供前端页面"""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """处理聊天请求（保留详细日志，方便调试）"""
    print("\n=== 收到前端聊天请求 ===")
    user_message = request.json.get('message', '')
    print(f"用户输入：{user_message}")
    try:
        response = agent.run(user_message)
        print(f"Agent响应：{response}")
        return jsonify({'response': response})
    except Exception as e:
        error_detail = f"处理错误：{str(e)}"
        print(error_detail)
        return jsonify({'response': f"抱歉，操作出错了：{str(e)}"}), 500


# ---------------------- 5. 运行入口（无修改） ----------------------
def run_agent_examples():
    print("📚 自习室预定系统Agent命令行版本\n")
    print("可输入以下指令测试：")
    print("- '测试预定系统' → 执行自动化预定")
    print("- '检查系统状态' → 验证系统是否可用")
    print("- '怎么预定自习室' → 获取帮助")
    print("- 输入'quit'退出\n")

    while True:
        user_input = input("你：")
        if user_input.lower() == 'quit':
            break
        try:
            response = agent.run(user_input)
            print(f"Agent：{response}\n")
        except Exception as e:
            print(f"Agent：抱歉，出错了：{str(e)}\n")


if __name__ == "__main__":
    # 检查命令行参数，启动Web服务或命令行版本
    if len(sys.argv) > 1 and sys.argv[1] == 'web':
        # 确保templates目录存在（避免前端页面找不到）
        os.makedirs('templates', exist_ok=True)
        # 复制index.html到templates目录（如果不存在）
        if not os.path.exists('templates/index.html'):
            with open('index.html', 'w', encoding='utf-8') as f:
                # 简单前端页面（保持原有逻辑）
                f.write("""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>图书馆自习室预定助手</title>
                    <style>
                        .container {width: 800px; margin: 50px auto; text-align: center;}
                        #messageInput {width: 600px; padding: 10px; font-size: 16px;}
                        #sendBtn {padding: 10px 20px; font-size: 16px;}
                        #chatHistory {margin-top: 30px; text-align: left; border: 1px solid #ccc; padding: 20px; height: 400px; overflow-y: auto;}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>图书馆自习室预定助手</h1>
                        <div id="chatHistory"></div>
                        <input type="text" id="messageInput" placeholder="输入指令（如'测试预定系统'）">
                        <button id="sendBtn">发送</button>
                    </div>
                    <script>
                        const chatHistory = document.getElementById('chatHistory');
                        const messageInput = document.getElementById('messageInput');
                        const sendBtn = document.getElementById('sendBtn');

                        // 发送消息
                        function sendMessage() {
                            const message = messageInput.value.trim();
                            if (!message) return;
                            // 添加用户消息到历史
                            chatHistory.innerHTML += `<p><strong>你：</strong>${message}</p>`;
                            messageInput.value = '';

                            // 调用后端API
                            fetch('/chat', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({message: message})
                            })
                            .then(res => res.json())
                            .then(data => {
                                // 添加Agent响应到历史
                                chatHistory.innerHTML += `<p><strong>助手：</strong>${data.response}</p>`;
                                // 滚动到底部
                                chatHistory.scrollTop = chatHistory.scrollHeight;
                            })
                            .catch(err => {
                                chatHistory.innerHTML += `<p><strong>助手：</strong>请求失败，请重试</p>`;
                            });
                        }

                        // 按钮点击发送
                        sendBtn.addEventListener('click', sendMessage);
                        // 回车发送
                        messageInput.addEventListener('keypress', e => {
                            if (e.key === 'Enter') sendMessage();
                        });
                    </script>
                </body>
                </html>
                """)
        # 启动Web服务
        print("🌐 Web服务已启动：http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        # 启动命令行版本
        run_agent_examples()