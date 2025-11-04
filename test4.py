from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC, expected_conditions
from selenium.webdriver.support.ui import Select
from time import sleep
import os


def automate_library_booking():
    """自动化图书馆自习室预定流程"""
    print("开始执行自习室预定自动化测试...")

    # 初始化Firefox浏览器
    driver = webdriver.Firefox()

    # 访问预定系统
    driver.get("https://booking.lib.hku.hk/Secure/FacilityStatusDate.aspx")
    print("已打开预定系统页面")

    # 等待页面加载
    sleep(2)

    # 查找用户名输入框并输入用户名
    print("正在输入用户名...")
    username_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "userid"))
    )
    username_field.clear()
    username_field.send_keys("u3665742")  # 替换为你的用户名

    # 等待页面加载
    sleep(2)

    # 查找密码输入框并输入密码
    print("正在输入密码...")
    password_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "password"))  # 假设密码框的name是"password"
    )
    password_field.clear()
    password_field.send_keys("Zjm20020808")  # 替换为你的密码

    # 等待页面加载
    sleep(2)

    # 点击登录按钮
    print("正在点击登录按钮...")
    # 这里需要根据实际的登录按钮选择器进行调整
    login_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH,
                                    "//input[@type='submit' or @type='button' or contains(@value, 'Login') or contains(@value, '登录')]"))
    )
    login_button.click()

    # 等待登录完成
    sleep(5)

    driver.find_element(By.ID, "main_ddlLibrary").click()
    sleep(2)
    # 尝试点击 "Main Library" 选项
    # 注意：这个XPath可能需要根据实际情况调整
    main_library_option = driver.find_element(By.XPATH,
                                              "/html/body/form/div[5]/div/div[1]/table/tbody/tr[1]/td[2]/select/option[6]")
    main_library_option.click()
    print("已尝试选择 Main Library")
    sleep(2)
    # 选择Computer设施类型
    driver.find_element(By.ID, "main_ddlType").click()
    sleep(2)
    driver.find_element(By.XPATH, "/html/body/form/div[5]/div/div[1]/table/tbody/tr[2]/td[2]/select/option[4]").click()
    sleep(2)
    # 选10月1日
    driver.find_element(By.ID, "main_ddlView").click()
    sleep(2)
    driver.find_element(By.XPATH, "/html/body/form/div[5]/div/div[1]/table/tbody/tr[3]/td[2]/select/option[3]").click()
    sleep(2)
    # 点击查询
    driver.find_element_by_id('main_btnGetResult').click()
    sleep(6)
    # 选择位置与时间
    driver.find_element(By.XPATH, "/html/body/form/div[5]/div/div[1]/div[4]/div/table/tbody/tr[2]/td[3]").click()
    sleep(2)
    # 提交预约
    driver.find_element(By.ID, "main_btnSubmit").click()
    sleep(2)
    # 确认预约(要用再加click)
    driver.find_element(By.ID, "main_btnSubmitYes")
    sleep(2)


# 调用函数执行自动化测试
if __name__ == "__main__":
    result = automate_library_booking()
    print("测试结果:", result)
