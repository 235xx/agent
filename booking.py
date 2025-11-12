from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import os
import warnings
import sys
from typing import Optional, List, Dict, Any
import requests
import json
from pydantic import Field

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage, LLMResult, Generation
from langchain.llms.base import LLM
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# 忽略LangChain弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")


# ---------------------- 1. 自定义ChatGLM LLM类（对接大模型API） ----------------------
class ChatGLM(LLM):
    api_url: str = Field(...)
    api_key: str = Field(...)

    def __init__(self, api_url: str, api_key: str, **kwargs):
        super().__init__(api_url=api_url, api_key=api_key, **kwargs)

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
            "temperature": 0.3,  # 降低随机性，确保格式稳定
            "max_tokens": 1000
        }
        if stop:
            data["stop"] = stop

        response = requests.post(self.api_url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"API请求失败: {response.status_code} - {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"] if (
                    "choices" in result and result["choices"]) else result.get("response", "")

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)


# 初始化ChatGLM（需确保API地址和密钥有效）
glm = ChatGLM(
    api_url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
    api_key="409c732b24c344eb9525919467821b13.Ep4NKHIocKvELO48"
)


# ---------------------- 2. 自习室预定自动化核心类 ----------------------
class StudyRoomBookingTester:
    def __init__(self):
        self.driver = None
        self.username = "u3665742"
        self.password = "Zjm20020808"

        # 场馆→option索引映射（需与页面实际一致）
        self.library_mapping = {
            "Chi Wah Learning Commons": 2,
            "Dental Library": 3,
            "Faculty of Machine": 4,
            "Law Library": 5,
            "main library": 6,
            "Medical Library": 7,
            "Music Library": 8,
            "Research Student Centre(Faculty of Engineering)": 9,
            "The University of Hong Kong History Gallery": 10,
        }

        # 场馆→{设施→option索引}映射（需与页面实际一致）
        self.library_facility_mapping = {
            "Chi Wah learning commons": {
                "study booth": 2,
                "study room": 3,
            },
            "Dental Library": {
                "Discussion Room": 2,
            },
            "Law Library": {
                "Discussion Room": 2,
                "Research Carrel(Higher Degree)": 3,
                "Study Table": 4,
            },
            "main library": {
                "AV Group Viewing Room": 2,
                "Communal Virtual PC": 3,
                "Computer": 4,
                "Concept and Creation Room": 5,
                "Discussion Room": 6,
                "Research Carrel(High Degree)": 7,
                "Single Study Room(3 sessions)": 8,
                "Study Table": 9,
                "Study Table(Deep Quiet)": 10,
            },
            "Medical Library": {
                "Discussion Room": 2,
                "Research Carrel(Higher Degree)": 3,
                "Single Study Room(Medical Library)": 4,
                "Software": 5,
            },
        }

        # 页面元素定位器
        self.login_btn_xpath = "//input[@type='submit' or @type='button' or contains(@value, 'Login') or contains(@value, '登录')]"
        self.library_select_id = "main_ddlLibrary"
        self.facility_select_id = "main_ddlType"
        self.date_select_id = "main_ddlView"
        self.date_option_xpath = "/html/body/form/div[5]/div/div[1]/table/tbody/tr[3]/td[2]/select/option[3]"
        self.query_btn_id = "main_btnGetResult"
        self.seat_xpath = "/html/body/form/div[5]/div/div[1]/div[4]/div/table/tbody/tr[2]/td[3]"
        self.submit_btn_id = " main_btnSubmit"
        self.confirm_btn_id = "main_btnSubmitYes"

    def _get_library_option_index(self, library_name: str) -> int:
        """根据场馆名称获取下拉框option索引"""
        normalized_name = library_name.strip().lower()
        for lib in self.library_mapping:
            if normalized_name in lib.lower():
                return self.library_mapping[lib]
        raise Exception(f"未找到场馆「{library_name}」，支持的场馆：{', '.join(self.library_mapping.keys())}")

    def _get_facility_option_index(self, library_name: str, facility_name: str) -> int:
        """根据场馆和设施名称获取下拉框option索引"""
        normalized_lib = library_name.strip().lower()
        normalized_facility = facility_name.strip().lower()

        for lib in self.library_facility_mapping:
            if normalized_lib in lib.lower():
                facility_map = self.library_facility_mapping[lib]
                if not facility_map:
                    raise Exception(f"场馆「{library_name}」暂无可用设施配置")
                for fac in facility_map:
                    if normalized_facility in fac.lower():
                        return facility_map[fac]
                available = ", ".join(facility_map.keys())
                raise Exception(f"场馆「{library_name}」没有「{facility_name}」，可用设施：{available}")

        raise Exception(f"未找到场馆「{library_name}」的设施配置")

    def run_booking_test(self, library_name: str, facility_name: str) -> str:
        """执行完整的自习室预定流程"""
        try:
            print(f"=== 开始预定：场馆={library_name}，设施={facility_name} ===")

            # 1. 初始化浏览器
            self.driver = webdriver.Firefox()
            self.driver.maximize_window()
            print("1. 浏览器初始化完成")

            # 2. 访问预定系统
            self.driver.get("https://booking.lib.hku.hk/Secure/FacilityStatusDate.aspx")
            print("2. 打开预定系统页面")
            sleep(2)

            # 3. 登录流程
            username_field = WebDriverWait(self.driver, 15).until(EC.presence_of_element_located((By.NAME, "userid")))
            username_field.clear()
            username_field.send_keys(self.username)
            sleep(2)

            password_field = WebDriverWait(self.driver, 15).until(EC.presence_of_element_located((By.NAME, "password")))
            password_field.clear()
            password_field.send_keys(self.password)
            sleep(2)

            login_button = WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable((By.XPATH, self.login_btn_xpath)))
            login_button.click()
            sleep(5)
            print("6. 登录成功")

            # 4. 选择场馆
            self.driver.find_element(By.ID, self.library_select_id).click()
            sleep(2)
            lib_index = self._get_library_option_index(library_name)
            library_xpath = f"/html/body/form/div[5]/div/div[1]/table/tbody/tr[1]/td[2]/select/option[{lib_index}]"
            self.driver.find_element(By.XPATH, library_xpath).click()
            print(f"7. 已选择场馆：{library_name}")
            sleep(3)

            # 5. 选择设施
            self.driver.find_element(By.ID, self.facility_select_id).click()
            sleep(2)
            fac_index = self._get_facility_option_index(library_name, facility_name)
            facility_xpath = f"/html/body/form/div[5]/div/div[1]/table/tbody/tr[2]/td[2]/select/option[{fac_index}]"
            self.driver.find_element(By.XPATH, facility_xpath).click()
            print(f"8. 已选择设施：{facility_name}")
            sleep(2)

            # 6. 选择日期
            self.driver.find_element(By.ID, self.date_select_id).click()
            sleep(2)
            self.driver.find_element(By.XPATH, self.date_option_xpath).click()
            print("9. 已选择日期")
            sleep(2)

            # 7. 查询座位
            self.driver.find_element(By.ID, self.query_btn_id).click()
            sleep(6)
            print("10. 座位数据加载完成")

            # 8. 选择座位
            target_seat = self.driver.find_element(By.XPATH, self.seat_xpath)
            target_seat.click()
            print("11. 已选择目标座位")
            sleep(2)

            # 9. 提交预约
            self.driver.find_element(By.ID, self.submit_btn_id).click()
            sleep(2)
            print("12. 预约请求已提交")

            # 10. 确认预约
            self.driver.find_element(By.ID, self.confirm_btn_id).click()
            sleep(3)
            print("13. 预约确认完成")

            return f"✅ 预定成功！已完成「{library_name}」的「{facility_name}」预约"

        except Exception as e:
            return f"❌ 预定失败：{str(e)}"

        finally:
            if self.driver:
                self.driver.quit()
                print("=== 浏览器已关闭 ===")


# ---------------------- 3. 工具函数与Agent初始化 ----------------------
tester = StudyRoomBookingTester()


def extract_library_facility(query: str) -> tuple:
    """从用户输入中提取场馆和设施（格式：预定[场馆]的[设施]）"""
    query = query.strip()
    for prefix in ["预定", "订", "帮我订", "我要订"]:
        if query.startswith(prefix):
            query = query[len(prefix):].strip()
    if "的" in query:
        parts = query.split("的", 1)
        return (parts[0].strip(), parts[1].strip())
    return (None, None)


def run_booking_tests(query):
    """工具函数：执行自习室预定"""
    library, facility = extract_library_facility(query)
    if not library or not facility:
        return "❌ 格式错误，请使用：'预定[场馆名称]的[设施名称]'（例如：预定Chi Wah Learning Commons的Study Booth）"
    return tester.run_booking_test(library_name=library, facility_name=facility)


def check_booking_status(query):
    """工具函数：检查系统状态"""
    try:
        driver = webdriver.Firefox()
        driver.get("https://booking.lib.hku.hk/Secure/FacilityStatusDate.aspx")
        sleep(3)
        status = "✅ 系统正常运行" if "Facility Status" in driver.title else "⚠️ 系统异常"
        driver.quit()
        return status
    except Exception as e:
        return f"❌ 系统检查失败：{str(e)}"


def get_booking_help(query):
    """工具函数：生成帮助信息"""
    libraries = "\n".join([f"- {lib}" for lib in tester.library_mapping.keys()])
    facilities = []
    for lib in tester.library_facility_mapping:
        if tester.library_facility_mapping[lib]:
            fac_list = ", ".join(tester.library_facility_mapping[lib].keys())
            facilities.append(f"- {lib}：{fac_list}")
    facilities_str = "\n".join(facilities) if facilities else "暂无配置设施"

    return f"""
    🏫 自习室预定帮助
    1. 支持的场馆：
    {libraries}
    2. 可用设施（按场馆分类）：
    {facilities_str}
    3. 预定格式示例：
       - 预定Chi Wah Learning Commons的Study Booth
       - 订Law Library的Discussion Room
    4. 其他功能：发送"检查系统状态"查看系统是否可用
    """


# 工具列表
tools = [
    Tool(
        name="RunBookingTests",
        func=run_booking_tests,
        description="用于预定自习室，需包含场馆和设施（格式：'预定[场馆]的[设施]'）"
    ),
    Tool(
        name="CheckBookingStatus",
        func=check_booking_status,
        description="查询预定系统是否正常运行（输入：'检查系统状态'）"
    ),
    Tool(
        name="GetBookingHelp",
        func=get_booking_help,
        description="获取支持的场馆、设施及预定格式（输入：'帮助'、'怎么预定'等）"
    )
]

# Agent系统提示
system_prompt = """你是自习室预定助手，严格按以下规则处理请求：
1. 若用户输入符合格式"预定[场馆]的[设施]"，直接调用RunBookingTests工具执行预定
2. 若格式错误，回复："请使用格式：'预定[场馆名称]的[设施名称]'（例如：预定Chi Wah Learning Commons的Study Booth）"
3. 若用户询问"帮助"、"支持哪些场馆"等，调用GetBookingHelp工具
4. 若用户询问"系统状态"、"系统能用吗"等，调用CheckBookingStatus工具
5. 不处理与预定无关的请求，回复："我仅支持自习室预定相关功能，发送'帮助'查看使用方法"
"""

# 初始化Agent
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(
    tools,
    glm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    agent_kwargs={"system_message": system_prompt},
    max_iterations=3  # 允许3次迭代确保工具调用完成
)

# ---------------------- 4. Flask Web服务 ----------------------
app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()
    try:
        response = agent.run(user_message)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f"处理失败：{str(e)}"}), 500


# ---------------------- 5. 运行入口 ----------------------
def run_agent_examples():
    print("📚 自习室预定系统Agent已启动\n")
    print("支持的指令示例：")
    print("- 预定Chi Wah Learning Commons的Study Booth")
    print("- 订Law Library的Discussion Room")
    print("- 帮助")
    print("- 检查系统状态")
    print("- 输入'quit'退出\n")

    while True:
        user_input = input("你：")
        if user_input.lower() == 'quit':
            break
        try:
            print(f"Agent：{agent.run(user_input)}\n")
        except Exception as e:
            print(f"Agent：处理出错：{str(e)}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'web':
        os.makedirs('templates', exist_ok=True)
        print("🌐 Web服务启动：http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        run_agent_examples()