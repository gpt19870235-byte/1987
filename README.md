# 程式內容檢查工具 - Windows EXE 產出

這個專案會用 GitHub Actions 自動把 `main0205.py` 打包成 **Windows 可直接執行的 .exe**
（目標電腦不用安裝 Python）。

## 你要做什麼（最簡單流程）
1. 在 GitHub 新建一個空 repo
2. 把本 repo 的檔案全部上傳（或用 git push）
3. 到 **Actions → Build Windows EXE → Run workflow**
4. 等跑完後，到該次 run 的 **Artifacts** 下載：
   - `program_checker_windows_exe_onefile`（單一 exe）
   - `program_checker_windows_exe_onedir`（資料夾版，啟動較快）

解壓後就可以在 Windows 直接開啟。

## 本機（Windows）自己打包（可選）
只要你的打包電腦有 Python：

```bat
pip install -r requirements.txt
pyinstaller --noconsole --onefile --name ProgramChecker main0205.py
```
產物在 `dist\ProgramChecker.exe`
