import joblib
import sys
import random

def parse_pkl_file(file_path):
    """
    Loads a .pkl file, prints its general information, 20 random keys,
    and all keys containing the substring 'sit'.
    """
    try:
        # 使用joblib加载.pkl文件
        all_data = joblib.load(file_path)
        
        print(f"Data type: {type(all_data)}")
        
        if isinstance(all_data, dict):
            num_keys = len(all_data)
            all_keys = list(all_data.keys()) # Get all keys once to reuse
            print(f"\nNumber of keys in the main dictionary: {num_keys}")
            
            # --- 功能1：随机打印20个key ---
            print("\n--- 20 Random Keys ---")
            num_to_sample = min(num_keys, 20)
            random_keys = random.sample(all_keys, num_to_sample)
            for i, key in enumerate(random_keys):
                print(f"  {i+1:2d}: {key}")
            
            # --- 新增功能：选出并打印包含 'sit' 的 key ---
            print("\n--- Keys containing 'sit' ---")
            
            # Use a list comprehension to filter keys containing 'sit' (case-insensitive)
            sit_keys = [key for key in all_keys if 'sit' in key.lower()]
            
            if sit_keys:
                print(f"Found {len(sit_keys)} keys containing 'sit':")
                for i, key in enumerate(sit_keys):
                    print(f"  {i+1}: {key}")
            else:
                print("No keys containing 'sit' were found.")
            # --- 功能结束 ---

            # 保留原有功能：分析第一个key作为示例
            print("\n--- Analysis of the first key (as an example) ---")
            first_key = all_keys[0]
            first_value = all_data[first_key]
            print(f"Key: '{first_key}'")
            
            if isinstance(first_value, dict):
                print("Keys of its nested dictionary:")
                for nested_key in first_value.keys():
                    print(f"  - {nested_key}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pkl_file_path = sys.argv[1]
        parse_pkl_file(pkl_file_path)
    else:
        print("Usage: python your_script_name.py <path_to_pkl_file>")