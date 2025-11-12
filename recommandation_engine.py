# recommandation_engine.py

import json
import requests
from typing import Dict, List, Optional
from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation
from pydantic import Field


# ============================================================
# 1. ChatGLM LLM 类
# ============================================================

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
            "model": "glm-4-flash",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2000
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


# ============================================================
# 2. 数据加载模块
# ============================================================

def load_menu_data(filename="tagged_restaurant.json"):
    """直接加载已有标签的数据"""
    print(f"📂 正在加载数据文件: {filename}")

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_items = 0
    total_restaurants = 0

    for restaurant in data.get("restaurants", []):
        if restaurant.get("actual_menu"):
            total_restaurants += 1
            for category in restaurant.get("actual_menu", []):
                total_items += len(category.get("items", []))

    print(f"✅ 成功加载 {total_restaurants} 家餐厅，共 {total_items} 道菜品")
    return data


# ============================================================
# 3. LLM 标签提取模块
# ============================================================

def extract_tags_with_llm(user_query: str, llm: ChatGLM) -> Dict:
    """使用LLM提取用户需求中的标签"""

    prompt = f"""请分析以下用户的餐饮需求，提取相关的标签信息。

用户需求：{user_query}

请严格按照以下JSON格式返回，不要添加任何其他文字说明：
{{
    "need_type": "food" 或 "drink" 或 "any",
    "keywords": [关键词列表],
    "cuisine_type": [可选：中式/港式/台式/韩式/日式/西式/意大利菜/融合料理/素食/有机餐],
    "food_category": [可选：咖啡饮品/茶类饮品/果汁/气泡饮/奶盖饮/奶茶/特饮/梳打/甜品/烘焙/轻食/三文治/汉堡/沙律/意粉/饭类/汤品/炸物/小食/套餐/下午茶餐],
    "flavor_profile": [可选：清淡/微甜/香浓/酸甜/辛辣/咸香/酥脆/奶香/果香/草本香/咖啡苦香/抹茶清香],
    "main_ingredients": [可选：鸡肉/猪肉/牛肉/鱼类/海鲜/豆制品/沙律蔬菜/乳制品/坚果/开心果/抹茶/茶叶/咖啡豆/谷物/螺丝粉/面类/蛋类],
    "diet_preference": [可选：有机/低糖/低脂/素食友好/无咖啡因选项/含咖啡因/手打饮品/无乳糖选项],
    "eating_scene": [可选：校园用餐/堂食/外带/下午茶/早餐/午餐/轻食时段/甜点时间],
    "price_range": [可选：低价（<HKD 20）/中低价（HKD 20–35）/中价（HKD 35–60）/中高价（HKD 60–90）/高价（>HKD 90）]
}}

分析规则：
1. need_type: 如果提到饮料、咖啡、茶、果汁等，设为"drink"；如果提到食物、饭、面、三明治等，设为"food"；否则设为"any"
2. keywords: 提取用户提到的所有关键词
3. 其他字段：根据用户需求匹配相应的标签，没有明确提到的保持为空列表
4. 价格：如果用户提到预算，转换为对应的价格区间

现在请分析用户需求并返回JSON："""

    try:
        response = llm._call(prompt)

        # 提取JSON部分
        response = response.strip()
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        # 解析JSON
        tags = json.loads(response)
        return tags
    except Exception as e:
        print(f"❌ 标签提取失败: {e}")
        print(f"LLM 返回: {response}")
        # 返回默认值
        return {
            "need_type": "any",
            "keywords": [],
            "cuisine_type": [],
            "food_category": [],
            "flavor_profile": [],
            "main_ingredients": [],
            "diet_preference": [],
            "eating_scene": [],
            "price_range": []
        }


# ============================================================
# 4. 菜品筛选模块
# ============================================================

def filter_items(menu_data: Dict, tags: Dict, strict_mode: bool = False) -> List[Dict]:
    """根据标签筛选菜品"""

    filtered_items = []

    # 类型映射
    type_mapping = {
        "food": ["炸物/小食", "轻食/三文治/汉堡", "沙律", "意粉/饭类", "汤品", "套餐/下午茶餐", "甜品/烘焙"],
        "drink": ["咖啡饮品", "茶类饮品", "果汁/气泡饮", "奶盖饮/奶茶", "特饮/梳打"]
    }

    for restaurant in menu_data.get("restaurants", []):
        if not restaurant.get("actual_menu"):
            continue

        for category in restaurant.get("actual_menu", []):
            for item in category.get("items", []):
                ai_tags = item.get("ai_tags", {})

                # 基础筛选：食物/饮品类型
                if tags["need_type"] != "any":
                    item_categories = ai_tags.get("food_category", [])
                    expected_categories = type_mapping.get(tags["need_type"], [])

                    # 检查菜品是否属于期望的类别
                    if not any(cat in item_categories for cat in expected_categories):
                        continue

                # 计算匹配分数
                match_score = 0
                total_criteria = 0

                # 1. 菜系匹配
                if tags.get("cuisine_type"):
                    total_criteria += 1
                    if any(cuisine in ai_tags.get("cuisine_type", []) for cuisine in tags["cuisine_type"]):
                        match_score += 1

                # 2. 食物类型匹配
                if tags.get("food_category"):
                    total_criteria += 1
                    if any(cat in ai_tags.get("food_category", []) for cat in tags["food_category"]):
                        match_score += 1

                # 3. 口味特征匹配
                if tags.get("flavor_profile"):
                    total_criteria += 1
                    if any(flavor in ai_tags.get("flavor_profile", []) for flavor in tags["flavor_profile"]):
                        match_score += 1

                # 4. 食材匹配
                if tags.get("main_ingredients"):
                    total_criteria += 1
                    if any(ing in ai_tags.get("main_ingredients", []) for ing in tags["main_ingredients"]):
                        match_score += 1

                # 5. 饮食偏好匹配
                if tags.get("diet_preference"):
                    total_criteria += 1
                    if any(pref in ai_tags.get("diet_preference", []) for pref in tags["diet_preference"]):
                        match_score += 1

                # 6. 场景标签匹配
                if tags.get("eating_scene"):
                    total_criteria += 1
                    if any(scene in ai_tags.get("eating_scene", []) for scene in tags["eating_scene"]):
                        match_score += 1

                # 7. 价格区间匹配
                if tags.get("price_range"):
                    total_criteria += 1
                    if ai_tags.get("price_range") in tags["price_range"]:
                        match_score += 1

                # 判断是否匹配
                if total_criteria == 0:
                    # 如果没有任何标签，返回所有菜品
                    filtered_items.append({
                        **item,
                        "match_score": 0,
                        "restaurant_name": restaurant.get("name", "")
                    })
                elif strict_mode:
                    # 严格模式：必须全部匹配
                    if match_score == total_criteria:
                        filtered_items.append({
                            **item,
                            "match_score": match_score,
                            "restaurant_name": restaurant.get("name", "")
                        })
                else:
                    # 宽松模式：至少匹配一个条件
                    if match_score > 0:
                        filtered_items.append({
                            **item,
                            "match_score": match_score,
                            "total_criteria": total_criteria,
                            "restaurant_name": restaurant.get("name", "")
                        })

    # 按匹配分数排序
    filtered_items.sort(key=lambda x: x["match_score"], reverse=True)

    return filtered_items


# ============================================================
# 5. 推荐生成模块
# ============================================================

def generate_recommendations(user_query: str, llm: ChatGLM, menu_data: Dict, top_n: int = 5) -> Dict:
    """生成推荐结果"""

    print("\n" + "=" * 60)
    print("🍽️  开始智能推荐")
    print("=" * 60)

    # 1. 提取标签
    print(f"\n🔍 分析用户需求: {user_query}")
    tags = extract_tags_with_llm(user_query, llm)
    print(f"✅ 提取的标签: {json.dumps(tags, ensure_ascii=False, indent=2)}")

    # 2. 先尝试宽松模式筛选
    print(f"\n🔎 筛选符合条件的菜品...")
    filtered_items = filter_items(menu_data, tags, strict_mode=False)

    print(f"✅ 找到 {len(filtered_items)} 道符合条件的菜品")

    # 3. 如果结果为空，尝试只按类型筛选
    if len(filtered_items) == 0 and tags["need_type"] != "any":
        print(f"\n⚠️  没有找到完全匹配的菜品，尝试只按类型筛选...")
        simple_tags = {
            "need_type": tags["need_type"],
            "keywords": tags["keywords"]
        }
        filtered_items = filter_items(menu_data, simple_tags, strict_mode=False)
        print(f"✅ 找到 {len(filtered_items)} 道符合类型的菜品")

    # 4. 返回前N个推荐
    recommendations = filtered_items[:top_n]

    if not recommendations:
        print("❌ 抱歉，没有找到符合条件的菜品。请尝试调整您的需求。")
        return {
            "query": user_query,
            "tags": tags,
            "recommendations": [],
            "total_found": 0
        }

    # 5. 格式化输出
    print(f"\n🌟 为您推荐以下 {len(recommendations)} 道菜品：\n")
    for i, item in enumerate(recommendations, 1):
        print(f"{i}. {item['name']}")
        print(f"   餐厅: {item['restaurant_name']}")
        print(f"   价格: {item.get('price', 'N/A')}")
        print(f"   匹配度: {'⭐' * int(item['match_score'])}")

        # 显示匹配的标签
        ai_tags = item.get("ai_tags", {})
        matched_tags = []
        if ai_tags.get("cuisine_type"):
            matched_tags.append(f"菜系: {', '.join(ai_tags['cuisine_type'])}")
        if ai_tags.get("flavor_profile"):
            matched_tags.append(f"口味: {', '.join(ai_tags['flavor_profile'])}")
        if ai_tags.get("food_category"):
            matched_tags.append(f"类型: {', '.join(ai_tags['food_category'])}")
        if matched_tags:
            print(f"   特点: {' | '.join(matched_tags)}")
        print()

    return {
        "query": user_query,
        "tags": tags,
        "recommendations": recommendations,
        "total_found": len(filtered_items)
    }


# ============================================================
# 6. 主程序
# ============================================================

def main():
    """主程序"""

    # 初始化 ChatGLM
    llm = ChatGLM(
        api_url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
        api_key="409c732b24c344eb9525919467821b13.Ep4NKHIocKvELO48"
    )

    # 加载菜单数据
    menu_data = load_menu_data("tagged_restaurants.json")

    # 测试用例
    test_queries = [
        "我想吃牛肉",
        "有没有什么清爽的饮品",
        "来点重口味食物",
        "我对麸质过敏，有没有可以吃的？"
    ]

    for query in test_queries:
        result = generate_recommendations(query, llm, menu_data, top_n=5)
        print("\n" + "~" * 60 + "\n")


if __name__ == "__main__":
    main()