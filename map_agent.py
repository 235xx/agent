import os
import json
import warnings
from time import sleep
from typing import Optional, List, Dict, Any, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain.llms.base import LLM
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from pydantic import Field
import requests

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

# ---------------------- 1) LLM: ChatGLM  ----------------------
class ChatGLM(LLM):
    api_url: str = Field(...)
    api_key: str = Field(...)

    def __init__(self, api_url: str, api_key: str, **kwargs):
        super().__init__(api_url=api_url, api_key=api_key, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "chatglm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": "glm-4.5",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,  # ✨ 提高创造性
            "max_tokens": 300,   # ✨ 减少 token 消耗
        }
        if stop:
            data["stop"] = stop
        
        try:
            resp = requests.post(self.api_url, headers=headers, json=data, timeout=10)
            resp.raise_for_status()
            js = resp.json()
            
            # 检查 API 错误
            if "error" in js:
                print(f"  [API 错误] {js['error']}")
                return ""
            
            if "choices" in js and js["choices"]:
                content = js["choices"][0]["message"]["content"]
                return content if content else ""
            
            return js.get("response", "")
        
        except requests.exceptions.Timeout:
            print(f"  [API 超时] 请求超过 10 秒")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"  [网络错误] {e}")
            return ""
        except Exception as e:
            print(f"  [未知错误] {e}")
            return ""


# 读取实体词表（多语言别名→官方英文）
BASE_DIR = os.path.dirname(__file__)
ENTITY_PATH = os.path.join(BASE_DIR, "entities.json")
with open(ENTITY_PATH, "r", encoding="utf-8") as f:
    ENTITIES = json.load(f)

# ✨ 新增：读取 facilities 子类别信息并合并到 ENTITIES
FACILITIES_PATH = os.path.join(BASE_DIR, "facilities.json")
FACILITIES = {}
try:
    with open(FACILITIES_PATH, "r", encoding="utf-8") as f:
        FACILITIES_DATA = json.load(f)
        FACILITIES = FACILITIES_DATA.get("facilities", {})
        
        # ✨ 将 facilities 数据合并到 ENTITIES 中（如果 entities.json 中没有）
        if "facilities" not in ENTITIES:
            ENTITIES["facilities"] = []
        
        # 添加所有 facility 到 ENTITIES（避免重复）
        existing_names = {item["name"] for item in ENTITIES.get("facilities", [])}
        for facility in FACILITIES.get("all", []):
            if facility["name"] not in existing_names:
                ENTITIES["facilities"].append({
                    "name": facility["name"],
                    "aliases": facility.get("aliases", []),
                    "type": "facility",
                    "subcategory": facility.get("subcategory", "")
                })
        
        print(f"[系统] 加载了 {len(FACILITIES.get('all', []))} 个设施")
except FileNotFoundError:
    print(f"[警告] 未找到 {FACILITIES_PATH}，facilities 子类别功能将不可用")
    FACILITIES = {}


# ---------------------- 2) 名称标准化与类别判断 ----------------------
SYSTEM_CANON_PROMPT = """
Generate ALL possible name variants for the location in user's query.
Return a JSON with keys: candidates, category, confidence.
- candidates: array of possible English names (official name, short name, common variants)
  Order by likelihood (most likely first)
- category: one of building, department, facility
- confidence: 0.0-1.0

Example:
Input: "张玉堂大楼在哪里？"
Output: {
  "candidates": ["Cheng Yu Tung Tower", "CYT Tower", "Cheng Yu Tung Building"],
  "category": "building",
  "confidence": 0.9
}

Only output JSON, no extra text.
"""

CANON_TEMPLATE = """
User query: {query}
Known entities: {aliases}
Generate all possible name variants.
"""


def generate_candidates_with_llm(llm: ChatGLM, query: str) -> Dict[str, Any]:
    """LLM 生成所有可能的名称候选（官方名、简称、变体）"""
    aliases = {
        "buildings": [{"name": b["name"], "aliases": b.get("aliases", [])} for b in ENTITIES.get("buildings", [])],
        "departments": [{"name": d["name"], "aliases": d.get("aliases", [])} for d in ENTITIES.get("departments", [])],
        "facilities": [{"name": f["name"], "aliases": f.get("aliases", [])} for f in ENTITIES.get("facilities", [])],
    }
    prompt = SYSTEM_CANON_PROMPT + "\n" + CANON_TEMPLATE.format(
        query=query, 
        aliases=json.dumps(aliases, ensure_ascii=False)
    )
    raw = llm._call(prompt)
    try:
        data = json.loads(raw) if raw else {}
        candidates = data.get("candidates", [])
        category = (data.get("category") or "").lower().strip()
        confidence = float(data.get("confidence") or 0.5)
        # 验证合法性
        if not candidates or category not in {"building", "department", "facility"}:
            name, cat = fallback_match(query)
            candidates = [name]
            category = cat
            confidence = 0.4
        # 去重并过滤空字符串
        candidates = [c.strip() for c in candidates if c and c.strip()]
        return {"candidates": candidates, "category": category, "confidence": confidence}
    except Exception as e:
        name, category = fallback_match(query)
        return {"candidates": [name], "category": category, "confidence": 0.3}


def local_match_exact(q: str) -> Optional[Tuple[str, str]]:
    """本地词表精确匹配：优先返回官方英文名"""
    ql = q.lower().strip()
    for cat in ("buildings", "departments", "facilities"):
        for item in ENTITIES.get(cat, []):
            # 精确匹配官方名称
            if item["name"].lower() == ql:
                cat_name = cat[:-1] if cat.endswith('s') else cat
                return item["name"], cat_name
            # 精确匹配任意别名
            for alias in item.get("aliases", []):
                if alias.lower() == ql:
                    cat_name = cat[:-1] if cat.endswith('s') else cat
                    return item["name"], cat_name
    return None


def local_match_fuzzy(q: str) -> Optional[Tuple[str, str]]:
    """模糊匹配：包含关系（用于问句提取关键词）"""
    ql = q.lower()
    for cat in ("buildings", "departments", "facilities"):
        for item in ENTITIES.get(cat, []):
            # 问句中包含官方名称
            if item["name"].lower() in ql:
                cat_name = cat[:-1] if cat.endswith('s') else cat
                return item["name"], cat_name
            # 问句中包含任意别名
            for alias in item.get("aliases", []):
                if alias.lower() in ql:
                    cat_name = cat[:-1] if cat.endswith('s') else cat
                    return item["name"], cat_name
    return None


def fallback_match(q: str) -> Tuple[str, str]:
    """兜底：先精确后模糊，最后返回原输入"""
    hit = local_match_exact(q)
    if hit:
        return hit
    hit = local_match_fuzzy(q)
    if hit:
        return hit
    # 默认猜测 building
    return q.strip(), "building"


# ---------------------- 新增：LLM 意图理解与关键词扩展 ----------------------

# 缓存常见查询的 LLM 结果（提高响应速度）
INTENT_CACHE = {
    "我想去运动": {"intent": "find_sports_facility", "keywords": ["运动", "体育", "sports", "gym", "fitness", "游泳", "swimming", "羽毛球", "篮球", "健身房"], "category_hint": "facility"},
    "我想吃饭": {"intent": "find_dining", "keywords": ["餐厅", "食堂", "canteen", "restaurant", "dining", "cafe", "咖啡", "美食", "吃饭"], "category_hint": "facility"},
    "我要学习": {"intent": "find_study_space", "keywords": ["图书馆", "library", "study", "自习室", "学习", "阅览室", "reading room"], "category_hint": "facility"},
    "我想运动": {"intent": "find_sports_facility", "keywords": ["运动", "体育", "sports", "gym", "fitness", "游泳", "swimming", "羽毛球", "篮球", "健身房"], "category_hint": "facility"},
    "哪里可以停车": {"intent": "find_parking", "keywords": ["parking", "停车", "泊车", "car park"], "category_hint": "facility"},
    "学校有银行吗": {"intent": "find_bank", "keywords": ["bank", "银行", "banking", "atm"], "category_hint": "facility"},
    "学校里有什么bank": {"intent": "find_bank", "keywords": ["bank", "银行", "banking", "atm"], "category_hint": "facility"},
}

def extract_intent_with_llm(llm: ChatGLM, query: str) -> Dict[str, Any]:
    """
    使用 LLM 理解用户意图并生成相关搜索词
    """
    # 检查缓存
    if query.strip() in INTENT_CACHE:
        return INTENT_CACHE[query.strip()]
    
    # ✨ 优化 prompt：更清晰的指令 + 更多示例
    prompt = f"""你是HKU校园导航助手。分析用户查询，返回JSON格式（不要其他文字）。

查询："{query}"

任务：识别意图并生成**中英文关键词**（包含同义词）

JSON格式：
{{
  "intent": "意图名称",
  "keywords": ["关键词1", "关键词2", ...],
  "category_hint": "building/department/facility"
}}

参考示例：

1️⃣ 功能类（生成相关设施关键词）：
"我想运动" → {{"intent":"find_sports","keywords":["运动","sports","gym","fitness","游泳","swimming"],"category_hint":"facility"}}
"哪里可以吃饭" → {{"intent":"find_dining","keywords":["餐厅","canteen","restaurant","食堂","cafe","dining"],"category_hint":"facility"}}
"学校有银行吗" → {{"intent":"find_bank","keywords":["bank","银行","banking","atm"],"category_hint":"facility"}}
"哪里可以停车" → {{"intent":"find_parking","keywords":["parking","停车","泊车","car park"],"category_hint":"facility"}}

2️⃣ 地点类（提取官方名称）：
"Main Building" → {{"intent":"find_place","keywords":["Main Building","main","大楼"],"category_hint":"building"}}
"图书馆" → {{"intent":"find_library","keywords":["Library","图书馆","library building"],"category_hint":"building"}}

现在处理："{query}"
仅返回JSON："""
    
    try:
        # ✨ 增加重试机制
        max_retries = 2
        for attempt in range(max_retries):
            try:
                raw = llm._call(prompt)
                
                # 检查是否为空
                if not raw or not raw.strip():
                    if attempt < max_retries - 1:
                        continue
                    raise ValueError("LLM 返回空响应")
                
                # 清理 markdown 标记
                raw = raw.strip()
                if "```" in raw:
                    # 提取 ``` 之间的内容
                    parts = raw.split("```")
                    for part in parts:
                        part = part.strip()
                        if part.startswith("json"):
                            part = part[4:].strip()
                        if part.startswith("{") and part.endswith("}"):
                            raw = part
                            break
                
                # 尝试解析 JSON
                data = json.loads(raw)
                
                # 验证必要字段
                if not data.get("keywords"):
                    if attempt < max_retries - 1:
                        continue
                    raise ValueError("JSON 缺少必要字段")
                
                if not data.get("category_hint") or data["category_hint"] not in {"building", "department", "facility"}:
                    data["category_hint"] = "facility"  # 默认设施
                
                result = {
                    "intent": data.get("intent", "unknown"),
                    "keywords": data.get("keywords", [query]),
                    "category_hint": data["category_hint"]
                }
                
                # 缓存结果
                INTENT_CACHE[query.strip()] = result
                
                return result
            
            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    continue
            except Exception:
                if attempt < max_retries - 1:
                    continue
        
        # ✨ 所有重试失败，使用规则兜底
        return fallback_intent_extraction(query)
    
    except Exception:
        return fallback_intent_extraction(query)


# ✨ 新增：规则兜底函数
def fallback_intent_extraction(query: str) -> Dict[str, Any]:
    """
    当 LLM 失败时使用规则提取意图
    """
    ql = query.lower()
    
    # 规则1：运动相关
    if any(kw in ql for kw in ["运动", "sport", "gym", "健身", "游泳", "羽毛球", "篮球", "跑步"]):
        return {
            "intent": "find_sports_facility",
            "keywords": ["运动", "体育", "sports", "gym", "fitness", "swimming", "游泳", "sport centre", "sports ground"],
            "category_hint": "facility"
        }
    
    # 规则2：休息相关
    if any(kw in ql for kw in ["休息", "rest", "座位", "lounge", "坐", "sitting", "relax"]):
        return {
            "intent": "find_rest_area",
            "keywords": ["休息", "rest", "lounge", "休息室", "common room", "座位", "sitting area", "student lounge"],
            "category_hint": "facility"
        }
    
    # 规则3：饮食相关
    if any(kw in ql for kw in ["吃", "饭", "餐", "canteen", "restaurant", "cafe", "咖啡", "食堂"]):
        return {
            "intent": "find_dining",
            "keywords": ["餐厅", "食堂", "canteen", "restaurant", "dining", "cafe", "咖啡", "coffee", "catering"],
            "category_hint": "facility",
            "subcategory": "Catering Outlets"
        }
    
    # 规则4：学习相关
    if any(kw in ql for kw in ["学习", "自习", "study", "library", "图书", "读书"]):
        return {
            "intent": "find_study_space",
            "keywords": ["图书馆", "library", "study", "自习室", "reading room", "学习空间"],
            "category_hint": "facility",
            "subcategory": "Libraries"
        }
    
    # 规则5：医疗相关
    if any(kw in ql for kw in ["医", "health", "clinic", "医疗", "诊所", "看病"]):
        return {
            "intent": "find_health_service",
            "keywords": ["health", "clinic", "medical", "医疗", "诊所", "health centre", "dental", "medical unit"],
            "category_hint": "facility",
            "subcategory": "Health Services"
        }
    
    # 规则6：打印相关
    if any(kw in ql for kw in ["打印", "print", "复印", "copy"]):
        return {
            "intent": "find_printing",
            "keywords": ["print", "打印", "printing", "copy", "复印", "computer", "computing"],
            "category_hint": "facility",
            "subcategory": "Computing Services"
        }
    
    # ✨ 规则7：停车相关
    if any(kw in ql for kw in ["停车", "parking", "泊车", "park", "车位"]):
        return {
            "intent": "find_parking",
            "keywords": ["parking", "停车", "泊车", "car park"],
            "category_hint": "facility",
            "subcategory": "Parking"
        }
    
    # ✨ 规则8：游泳相关
    if any(kw in ql for kw in ["游泳", "swimming", "pool", "游泳池"]):
        return {
            "intent": "find_swimming",
            "keywords": ["swimming", "游泳", "pool", "游泳池"],
            "category_hint": "facility",
            "subcategory": "Sports"
        }
    
    # ✨ 规则9：厕所相关
    if any(kw in ql for kw in ["厕所", "toilet", "washroom", "restroom", "洗手间", "卫生间"]):
        return {
            "intent": "find_toilet",
            "keywords": ["toilet", "厕所", "washroom", "restroom", "洗手间"],
            "category_hint": "facility"
        }
    
    # ✨ 规则10：银行相关
    if any(kw in ql for kw in ["银行", "bank", "atm", "取钱", "存钱"]):
        return {
            "intent": "find_bank",
            "keywords": ["bank", "银行", "banking", "atm"],
            "category_hint": "facility",
            "subcategory": "Banking Services"
        }
    
    # 默认：按原查询搜索
    return {
        "intent": "unknown",
        "keywords": [query, query.replace("？", "").replace("?", "").strip()],
        "category_hint": "facility"
    }


def search_by_keywords(keywords: List[str], subcategory: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """
    使用多个关键词在本地词表中搜索
    返回：[(官方名称, 类别, 匹配的关键词), ...]
    
    如果指定了 subcategory，会优先搜索 facilities.json 中的该子类别
    """
    results = []
    seen = set()  # 去重
    
    # ✨ 优先在 facilities 子类别中搜索
    if subcategory and FACILITIES:
        subcategory_items = FACILITIES.get("subcategory", {}).get(subcategory, [])
        for item in subcategory_items:
            official_name = item["name"]
            if official_name not in seen:
                # 使用第一个关键词作为匹配关键词
                results.append((official_name, "facility", keywords[0] if keywords else ""))
                seen.add(official_name)
        
        # 如果在子类别中找到了结果，直接返回
        if results:
            return results
    
    # 常规搜索
    for keyword in keywords:
        # 对每个关键词进行模糊匹配
        kw_lower = keyword.lower().strip()
        if not kw_lower:
            continue
        
        for cat in ("buildings", "departments", "facilities"):
            for item in ENTITIES.get(cat, []):
                cat_name = cat[:-1] if cat.endswith('s') else cat
                official_name = item["name"]
                
                # 跳过重复项
                if official_name in seen:
                    continue
                
                # 检查官方名称是否包含关键词
                if kw_lower in official_name.lower():
                    results.append((official_name, cat_name, keyword))
                    seen.add(official_name)
                    continue
                
                # 检查别名是否包含关键词
                for alias in item.get("aliases", []):
                    if kw_lower in alias.lower():
                        results.append((official_name, cat_name, keyword))
                        seen.add(official_name)
                        break
    
    return results


# ---------------------- 3) Selenium 抓取：根据类别执行左栏搜索/点击 ----------------------
class HKUMapClient:
    def __init__(self):
        self.driver = None

    def _ensure(self):
        if self.driver is None:
            self.driver = webdriver.Firefox()
            self.driver.get("http://www.maps.hku.hk/")
            sleep(2)

    def _click_first_match_in_list(self, list_el, name: str) -> bool:
        items = list_el.find_elements(By.CSS_SELECTOR, "a,li")
        name_l = name.lower()
        # 优先全包含匹配，其次任意子串匹配
        for it in items:
            t = it.text.strip()
            if not t:
                continue
            if t.lower() == name_l or name_l in t.lower():
                it.click()
                return True
        return False

    def _expand_and_click_facility(self, name: str, subcategory: Optional[str] = None) -> bool:
        """
        在 Facilities 标签页中展开并点击设施
        
        如果指定了 subcategory，会直接点击该子类别，然后在展开的列表中查找设施
        """
        # 切到 Facilities 页签
        try:
            self.driver.find_element(By.ID, "navmenutab_Facilities").click()
            sleep(0.4)
        except Exception as e:
            return False

        # ✨ 准备名称变体（支持多种格式匹配）
        name_l = name.lower()
        name_variants = [
            name_l,  # 原始名称
            name_l.replace(":", ""),  # 移除冒号
            name_l.replace(":", " "),  # 冒号替换为空格
            name_l.replace("bldg", "building"),  # Bldg → Building
            name_l.replace("building", "bldg"),  # Building → Bldg
        ]
        
        # 如果名称中包含冒号，也尝试匹配冒号后的部分（如 "Convenient store: 7-ELEVEN" → "7-ELEVEN"）
        if ":" in name_l:
            parts = name_l.split(":")
            if len(parts) > 1:
                name_variants.append(parts[-1].strip())  # 取冒号后的部分
        
        # ✨ 如果指定了子类别，先点击该子类别
        subcategory_clicked = False
        if subcategory:
            try:
                # 在页面中查找包含子类别文本的元素
                subcategory_elements = self.driver.find_elements(By.TAG_NAME, "td")
                for elem in subcategory_elements:
                    if elem.text.strip() == subcategory:
                        elem.click()
                        sleep(0.5)  # 等待列表展开
                        subcategory_clicked = True
                        break
            except Exception:
                pass
        
        # ✨ 如果成功点击了子类别，在展开的列表中查找设施
        if subcategory_clicked:
            try:
                # 查找展开的子类别列表
                sleep(0.3)  # 额外等待确保列表完全展开
                items = self.driver.find_elements(By.CSS_SELECTOR, "a, li")
                for it in items:
                    t = it.text.strip()
                    if not t:
                        continue
                    t_lower = t.lower()
                    
                    # 尝试所有名称变体
                    for variant in name_variants:
                        if variant in t_lower or t_lower in variant:
                            it.click()
                            return True
            except Exception:
                pass
        
        # ✨ 兜底：遍历所有设施类目（原有逻辑）
        cats = self.driver.find_elements(By.CSS_SELECTOR, "[id^='NavMenu-facilities-']")
        cats = [c for c in cats if not c.get_attribute("id").endswith("-sublist")]

        for c in cats:
            try:
                c.click()
                sleep(0.2)
                sub_id = c.get_attribute("id") + "-sublist"
                sub = self.driver.find_element(By.ID, sub_id)
                # 在子列表中按文本模糊查找
                items = sub.find_elements(By.CSS_SELECTOR, "a,li")
                for it in items:
                    t = it.text.strip()
                    if not t:
                        continue
                    t_lower = t.lower()
                    
                    # 尝试所有名称变体
                    for variant in name_variants:
                        if variant in t_lower or t_lower in variant:
                            it.click()
                            return True
            except Exception:
                continue
        return False

    def _search_in_box_and_click(self, box_id: str, list_id: str, name: str) -> bool:
        wait = WebDriverWait(self.driver, 10)
        box = wait.until(EC.presence_of_element_located((By.ID, box_id)))
        box.clear()
        box.send_keys(name)
        sleep(0.3)
        # 有些站点依赖回车触发过滤
        box.send_keys(Keys.ENTER)
        sleep(0.6)
        lst = wait.until(EC.presence_of_element_located((By.ID, list_id)))
        return self._click_first_match_in_list(lst, name)

    def query_location(self, name: str, category: str, subcategory: Optional[str] = None) -> str:
        """
        building: 默认在 Buildings 标签页，使用 #buildingsearchbox + #buildinglist
        department: 需要先点击 Departments 标签页，然后使用 #departmentsearchbox + #departmentlist
        facility: 点击 Facilities 标签页，如果指定了 subcategory 则点击子类别
        返回：(是否成功, 结果消息)
        """
        try:
            self._ensure()
            if category == "building":
                # ✨ 确保切换到 Buildings 标签页（第一个标签）
                try:
                    bldg_tab = self.driver.find_element(By.CSS_SELECTOR, 'a.tab[rel="#tab_1_contents"]')
                    bldg_tab.click()
                    sleep(0.5)
                except Exception:
                    pass
                
                # 在 Buildings 搜索框中搜索
                ok = self._search_in_box_and_click("buildingsearchbox", "buildinglist", name)
                return (ok, f"Building: {name}")
            elif category == "department":
                # 先切换到 Departments 标签页
                try:
                    dept_tab = self.driver.find_element(By.CSS_SELECTOR, 'a.tab[rel="#tab_2_contents"]')
                    dept_tab.click()
                    sleep(0.5)
                except Exception:
                    pass
                
                # 在 Departments 搜索框中搜索
                ok = self._search_in_box_and_click("departmentsearchbox", "departmentlist", name)
                return (ok, f"Department: {name}")
            else:  # facility
                ok = self._expand_and_click_facility(name, subcategory)
                return (ok, f"Facility: {name}")
        except Exception as e:
            return (False, f"查询失败：{e}")

    def query_location_with_candidates(self, candidates: List[str], category: str, subcategory: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
        """
        逐一尝试候选名称，返回 (是否成功, 结果消息, 成功的名称)
        """
        for name in candidates:
            success, msg = self.query_location(name, category, subcategory)
            if success:
                return (True, msg, name)
        return (False, f"所有候选均未找到: {', '.join(candidates)}", None)

    def _first_result_text(self, selector: str) -> str:
        try:
            elems = self.driver.find_elements(By.CSS_SELECTOR, selector)
            return elems[0].text if elems else "无结果"
        except Exception:
            return "无结果"

    def _panel_text(self, selector: str) -> str:
        try:
            el = self.driver.find_element(By.CSS_SELECTOR, selector)
            return el.text
        except Exception:
            return "未获取到详情"

    def close(self):
        if self.driver:
            self.driver.quit()
            self.driver = None


MAP = HKUMapClient()


# ---------------------- 4) LangChain 工具与 Agent ----------------------

# ---------------------- 新增：相似度匹配函数 ----------------------
from difflib import SequenceMatcher

def calculate_similarity(s1: str, s2: str) -> float:
    """计算两个字符串的相似度 (0.0-1.0)"""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def find_best_matches(query: str, top_n: int = 3) -> List[Tuple[str, str, float]]:
    """
    在本地词表中找到最相似的 top_n 个结果
    返回: [(官方名称, 类别, 相似度分数), ...]
    """
    ql = query.lower().strip()
    candidates = []
    
    for cat in ("buildings", "departments", "facilities"):
        for item in ENTITIES.get(cat, []):
            cat_name = cat[:-1] if cat.endswith('s') else cat
            
            # 计算与官方名称的相似度
            score = calculate_similarity(ql, item["name"])
            candidates.append((item["name"], cat_name, score))
            
            # 计算与所有别名的相似度
            for alias in item.get("aliases", []):
                alias_score = calculate_similarity(ql, alias)
                if alias_score > score:  # 取最高分
                    score = alias_score
                    candidates[-1] = (item["name"], cat_name, score)
    
    # 按相似度降序排序，取前 top_n
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:top_n]


# 全局变量：用于存储待确认的候选
pending_confirmation = {"candidates": [], "query": ""}

def tool_query_location(q: str) -> str:
    """
    增强版搜索流程：
    1. 精确匹配 → 直接搜索
    2. 模糊匹配 → 直接搜索
    3. LLM 语义理解 → 生成关键词 → 批量搜索 → 返回候选列表
    4. 相似度匹配 → 返回候选列表
    """
    # 步骤1：精确匹配（完全一致）
    local_hit = local_match_exact(q)
    if local_hit:
        name, category = local_hit
        success, msg, _ = MAP.query_location_with_candidates([name], category)
        if success:
            return f"✓ 已为您找到：{name}（{category}）"
        return f"⚠ 词表中有 {name}，但地图未能定位"
    
    # 步骤2：模糊匹配（包含关系）
    fuzzy_hit = local_match_fuzzy(q)
    if fuzzy_hit:
        name, category = fuzzy_hit
        success, msg, _ = MAP.query_location_with_candidates([name], category)
        if success:
            return f"✓ 已为您找到：{name}（{category}）"
    
    # ✨ 步骤3：LLM 语义理解（新增）
    # 判断是否需要 LLM（如果查询很短或很模糊）
    if len(q.strip()) < 15 and not any(char.isdigit() for char in q):
        intent_data = extract_intent_with_llm(glm, q)
        keywords = intent_data["keywords"]
        category_hint = intent_data["category_hint"]
        subcategory = intent_data.get("subcategory")  # ✨ 获取子类别
        
        # 使用关键词批量搜索
        search_results = search_by_keywords(keywords, subcategory)
        
        if search_results:
            # 按类别过滤（优先匹配 LLM 建议的类别）
            filtered = [r for r in search_results if r[1] == category_hint]
            if not filtered:
                filtered = search_results  # 没有匹配类别时使用全部结果
            
            # 限制返回数量
            filtered = filtered[:5]
            
            # 返回候选列表（JSON格式）
            candidates_info = [
                {
                    "name": name,
                    "category": cat,
                    "matched_keyword": keyword,
                    "score": 0.75,  # LLM 匹配给予较高分数
                    "subcategory": subcategory  # ✨ 传递子类别信息
                }
                for name, cat, keyword in filtered
            ]
            return json.dumps({
                "type": "location_candidates",
                "content": candidates_info
            }, ensure_ascii=False)
    
    # 步骤4：相似度匹配
    matches = find_best_matches(q, top_n=3)
    if matches and matches[0][2] > 0.6:  # 相似度 > 0.6 直接搜索
        name, category, score = matches[0]
        success, msg, _ = MAP.query_location_with_candidates([name], category)
        if success:
            return f"✓ 已为您找到：{name}（{category}）"
    
    # 步骤5：返回候选列表（需要确认）
    if matches and matches[0][2] > 0.3:
        candidates_info = [{"name": m[0], "category": m[1], "score": m[2]} for m in matches]
        return json.dumps({
            "type": "location_confirm",
            "content": candidates_info
        }, ensure_ascii=False)
    
    # 步骤6：完全无匹配
    return json.dumps({
        "type": "location",
        "content": f"抱歉，未能找到与「{q}」相关的地点。\n请尝试使用建筑物/部门的官方名称或常用简称。"
    }, ensure_ascii=False)


def handle_user_query(q: str) -> str:
    """
    在 Agent 外层处理确认流程（支持 LLM 多候选结果）
    """
    global pending_confirmation
    
    # 检查是否有待确认的候选
    if pending_confirmation["candidates"]:
        # 用户回复"是"
        if q.strip().lower() in {"是", "yes", "对", "y", "确认", "1"}:
            name, category = pending_confirmation["candidates"][0][:2]
            subcategory = pending_confirmation.get("subcategory")  # ✨ 获取子类别
            pending_confirmation = {"candidates": [], "query": ""}
            success, msg, _ = MAP.query_location_with_candidates([name], category, subcategory)
            if success:
                return f"✓ 已为您找到：{name}（{category}）"
            return f"⚠ 抱歉，未能在地图上定位到 {name}"
        
        # 用户回复"否"
        elif q.strip().lower() in {"否", "no", "不是", "n", "0"}:
            pending_confirmation = {"candidates": [], "query": ""}
            return "好的，请重新描述您要找的地点。"
        
        # 用户选择其他候选（2-5）
        elif q.strip() in {"2", "3", "4", "5"}:
            idx = int(q.strip()) - 1
            if idx < len(pending_confirmation["candidates"]):
                name, category = pending_confirmation["candidates"][idx][:2]
                subcategory = pending_confirmation.get("subcategory")  # ✨ 获取子类别
                pending_confirmation = {"candidates": [], "query": ""}
                success, msg, _ = MAP.query_location_with_candidates([name], category, subcategory)
                if success:
                    return f"✓ 已为您找到：{name}（{category}）"
                return f"⚠ 抱歉，未能在地图上定位到 {name}"
    
    # 调用工具进行搜索
    result = tool_query_location(q)
    
    # ✨ 处理 LLM 返回的多个结果
    if result.startswith("LLM_RESULTS:"):
        candidates_json = result.replace("LLM_RESULTS:", "")
        candidates_info = json.loads(candidates_json)
        
        # 转换为内部格式
        pending_confirmation["candidates"] = [(c["name"], c["category"], c["score"]) for c in candidates_info]
        pending_confirmation["query"] = q
        pending_confirmation["subcategory"] = candidates_info[0].get("subcategory") if candidates_info else None  # ✨ 保存子类别
        
        # 生成确认提示（显示匹配关键词）
        suggestions = "\n".join([
            f"  {i+1}. {c['name']} ({c['category']}) - 匹配关键词: {c.get('matched_keyword', 'N/A')}" 
            for i, c in enumerate(candidates_info)
        ])
        
        return (f"🔍 根据您的需求「{q}」，找到以下相关地点：\n{suggestions}\n\n"
                f"请回复数字（1-{len(candidates_info)}）选择，或回复「否」重新输入。")
    
    # 处理相似度匹配的确认
    if result.startswith("NEED_CONFIRM:"):
        candidates_json = result.replace("NEED_CONFIRM:", "")
        candidates_info = json.loads(candidates_json)
        
        # 转换为内部格式
        pending_confirmation["candidates"] = [(c["name"], c["category"], c["score"]) for c in candidates_info]
        pending_confirmation["query"] = q
        
        # 生成确认提示
        suggestions = "\n".join([
            f"  {i+1}. {c['name']} ({c['category']}) - 相似度 {c['score']:.0%}" 
            for i, c in enumerate(candidates_info)
        ])
        
        return (f"未找到完全匹配的结果，以下是最接近的选项：\n{suggestions}\n\n"
                f"请问您要找的是「{candidates_info[0]['name']}」吗？\n"
                f"回复「1」或「是」选择第一个，「2」-「{len(candidates_info)}」选择其他选项，「否」重新输入。")
    
    return result


tools = [
    Tool(name="HKUMapQuery", func=tool_query_location, description="Query HKU map for building/department/facility location by natural language."),
]

SYSTEM_PROMPT = (
    "你是HKU地图助手。用户可能用中文或其他语言询问地点。"
    "你需要先将地点名称标准化为官方英文，并判断类别（building/department/facility），"
    "然后调用 HKUMapQuery 工具执行搜索（building/department 用搜索框；facility 用点击）。"
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 初始化 LLM（用占位Key，用户自行替换）
glm = ChatGLM(
    api_url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
    api_key=os.environ.get("BIGMODEL_API_KEY", "409c732b24c344eb9525919467821b13.Ep4NKHIocKvELO48")
)

agent = initialize_agent(
    tools,
    glm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    agent_kwargs={"system_message": SYSTEM_PROMPT}
)


# ---------------------- 5) Flask 交互端口 ----------------------
app = Flask(__name__)
CORS(app)


# 修改原/chat路由为/map_chat
@app.route("/map_chat", methods=["POST"])
def chat():
    msg = request.json.get("message", "")
    try:
        resp = handle_user_query(msg)
        return jsonify({"response": resp})
    except Exception as e:
        return jsonify({"response": f"错误：{e}"}), 500

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        print("Map Agent 服务启动：http://localhost:5000")
        app.run(host="0.0.0.0", port=5000, debug=True)
    else:
        # 命令行交互（使用新的处理函数）
        print("HKU 地图助手已启动！输入 'exit' 或 'quit' 退出。")
        print("=" * 60)
        try:
            while True:
                q = input("\n你：")
                if q.strip().lower() in {"exit", "quit"}:
                    break
                # 使用新的处理函数，支持确认流程
                response = handle_user_query(q)
                print(f"Agent：{response}")
        finally:
            MAP.close()
