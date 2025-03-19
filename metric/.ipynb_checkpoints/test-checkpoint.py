import os

def rename_files(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件名是否以 'restoration_' 开头
        if filename.startswith('seg_'):
            # 构建新的文件名
            new_filename = filename.replace('seg_', '', 1)
            # 获取完整的文件路径
            src = os.path.join(folder_path, filename)
            dst = os.path.join(folder_path, new_filename)
            # 重命名文件
            os.rename(src, dst)
            print(f'Renamed: {filename} -> {new_filename}')

if __name__ == "__main__":
    # 替换为你的文件夹路径，例如 "C:/Users/你的用户名/图片"
    folder_path = "test1"
    rename_files(folder_path)
