import os
import ast
import sys

def create_dir(directory):
    if os.path.isdir(directory):
        return directory
    print(f"creating new dir: {directory}")
    os.makedirs(directory)
    return directory
       
def append_to_file(file_path, line):
    with open(file_path, 'a') as file:
            file.write(line) 

def create_saved_models_dir(self):
        d = "saved_models"
        if not os.path.isdir(d):
            print(f"creating new dir: {d}")
        full_path = os.path.join(d, f"run_{self.run_number}")
        if not os.path.isdir(full_path):
            os.makedirs(full_path)
        return full_path

def log_summary(output_file, model):
    with open(output_file, "w") as f:
        out = sys.stdout
        sys.stdout = f
        model.summary()
        sys.stdout = out

def log_model(output_file, is_transfer_learning):
    model_file_name = "model"
    if is_transfer_learning:
        model_file_name = "transfer_model"
    source_file = os.path.join("code", "model", "models", f"{model_file_name}.py")
    with open(source_file, "r") as f:
        source_content = f.read()
    tree = ast.parse(source_content)
    imports = []
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            imports.append(ast.unparse(node))        
        elif isinstance(node, ast.ClassDef):
            class_code = ast.unparse(node)
            classes.append(class_code)
    with open(output_file, "w") as f:
        for imp in imports:
            f.write(imp + '\n')
        f.write("\n\n".join(classes))
        
        
def log_hyperparameters(output_file):
    source_file = os.path.join("code", "model", "hyperparameters.py")
    with open(source_file, "r") as f:
        source_content = f.read()
    tree = ast.parse(source_content)
    global_vars = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    try:
                        var_value = ast.literal_eval(node.value)
                        global_vars[var_name] = var_value
                    except (ValueError, TypeError):
                        print(f"Skipping non-literal value assignment: {var_name}")
    with open(output_file, "w") as f:
        for var_name, var_value in global_vars.items():
            f.write(f"{var_name} = {var_value}\n")
            
            
def get_op(node):
    if isinstance(node, ast.Div):
        return "/"
    return None

def evaluate_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.Str):
        return node.s
    elif isinstance(node, ast.NameConstant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left, right = node.left.value, node.right.value
        op = get_op(node.op)
        return str(left) + op + str(right)            
    else:
        return ""

def log_data_augmentation(output_file):
    def get_image_data_generator_params(tree):
        idg_params = {}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == "get_train_data"):
                continue
            for item in ast.walk(node):
                if not (isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr == "ImageDataGenerator"):
                    continue 
                for keyword in item.keywords:
                    param_value = evaluate_node(keyword.value)
                    if param_value is not None:
                        idg_params[keyword.arg] = param_value
                return idg_params
        return None        
    source_file = os.path.join("code", "model", "dataset.py")
    with open(source_file, "r") as f:
        source_content = f.read()
    t = ast.parse(source_content)
    params = get_image_data_generator_params(t)
    if params is None:
        print("error: No data from image data generator. Not creating file")
        return
    with open(output_file, "w") as f:
        for param_name, param_value in params.items():
            f.write(f"{param_name} = {param_value},\n")
