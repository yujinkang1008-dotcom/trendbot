"""
Trendbot 실행 스크립트
"""
import os
import sys

# 프로젝트 루트 디렉터리를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Streamlit 앱 실행
if __name__ == "__main__":
    import subprocess
    import sys
    
    # Streamlit 앱 실행
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        os.path.join(project_root, "src", "main.py"),
        "--server.port", "8501",
        "--server.address", "localhost"
    ])
