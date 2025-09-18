#!/usr/bin/env python3
"""
Simple utility to verify the ACDC portfolio installation and setup.
"""

import sys
import os
import importlib
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_required_packages():
    """Check if required packages are installed"""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy',
        'opencv-python', 'pillow', 'scikit-image', 'scikit-learn',
        'torch', 'torchvision', 'jupyter', 'tqdm', 'pyyaml'
    ]
    
    optional_packages = [
        'open3d', 'pyntcloud', 'tensorflow', 'paho-mqtt', 'websockets'
    ]
    
    missing_required = []
    missing_optional = []
    
    print("\nChecking required packages:")
    for package in required_packages:
        try:
            # Handle special cases
            if package == 'opencv-python':
                importlib.import_module('cv2')
            elif package == 'pillow':
                importlib.import_module('PIL')
            elif package == 'scikit-image':
                importlib.import_module('skimage')
            elif package == 'scikit-learn':
                importlib.import_module('sklearn')
            elif package == 'pyyaml':
                importlib.import_module('yaml')
            else:
                importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_required.append(package)
    
    print("\nChecking optional packages:")
    for package in optional_packages:
        try:
            if package == 'paho-mqtt':
                importlib.import_module('paho.mqtt.client')
            else:
                importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ⚠️  {package} (optional)")
            missing_optional.append(package)
    
    return missing_required, missing_optional

def check_jupyter_installation():
    """Check if Jupyter is properly installed"""
    try:
        result = subprocess.run(['jupyter', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Jupyter Notebook is installed")
            return True
        else:
            print("❌ Jupyter installation issue")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Jupyter not found in PATH")
        return False

def check_portfolio_structure():
    """Check if portfolio directory structure is complete"""
    expected_dirs = [
        '01_image_segmentation',
        '02_point_cloud_processing', 
        '03_object_tracking',
        '04_occupancy_mapping',
        '05_vehicle_guidance',
        '06_v2x_communication'
    ]
    
    expected_subdirs = ['notebooks', 'src', 'data', 'docs', 'examples']
    
    print("\nChecking portfolio structure:")
    missing_dirs = []
    
    for module_dir in expected_dirs:
        if os.path.exists(module_dir):
            print(f"  ✅ {module_dir}/")
            
            # Check subdirectories
            for subdir in expected_subdirs:
                full_path = os.path.join(module_dir, subdir)
                if os.path.exists(full_path):
                    print(f"    ✅ {subdir}/")
                else:
                    print(f"    ⚠️  {subdir}/ (empty)")
        else:
            print(f"  ❌ {module_dir}/")
            missing_dirs.append(module_dir)
    
    return missing_dirs

def check_notebooks():
    """Check if example notebooks exist"""
    notebook_examples = [
        '01_image_segmentation/notebooks/01_semantic_segmentation_basics.ipynb',
        '03_object_tracking/notebooks/01_tracking_fundamentals.ipynb',
        '06_v2x_communication/notebooks/01_v2x_fundamentals.ipynb'
    ]
    
    print("\nChecking example notebooks:")
    missing_notebooks = []
    
    for notebook in notebook_examples:
        if os.path.exists(notebook):
            print(f"  ✅ {os.path.basename(notebook)}")
        else:
            print(f"  ❌ {notebook}")
            missing_notebooks.append(notebook)
    
    return missing_notebooks

def provide_setup_instructions(missing_required, missing_optional, missing_dirs, missing_notebooks):
    """Provide setup instructions for missing components"""
    
    if missing_required:
        print("\n🔧 SETUP INSTRUCTIONS")
        print("=" * 50)
        print("\n1. Install missing required packages:")
        print("   pip install", " ".join(missing_required))
    
    if missing_optional:
        print("\n2. Install optional packages (recommended):")
        print("   pip install", " ".join(missing_optional))
    
    if missing_dirs:
        print("\n3. Missing directories detected. You may need to:")
        print("   - Re-clone the repository")
        print("   - Check if you're in the correct directory")
        
    if missing_notebooks:
        print("\n4. Some example notebooks are missing.")
        print("   - This is normal if you're just starting")
        print("   - Create notebooks as you work through the modules")
    
    print("\n5. To start using the portfolio:")
    print("   jupyter notebook")
    print("   # Then navigate to any module's notebooks/ directory")

def main():
    """Main setup verification function"""
    print("ACDC Practical Exercises - Portfolio Setup Verification")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup verification failed: Python version incompatible")
        return False
    
    # Check packages
    missing_required, missing_optional = check_required_packages()
    
    # Check Jupyter
    jupyter_ok = check_jupyter_installation()
    
    # Check structure
    missing_dirs = check_portfolio_structure()
    
    # Check notebooks
    missing_notebooks = check_notebooks()
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP VERIFICATION SUMMARY")
    print("=" * 60)
    
    if not missing_required and jupyter_ok and not missing_dirs:
        print("🎉 Portfolio setup is ready!")
        print("\nTo get started:")
        print("  1. Navigate to any module directory")
        print("  2. Run: jupyter notebook")
        print("  3. Open the example notebooks")
        
        if missing_optional:
            print(f"\n⚠️  {len(missing_optional)} optional packages missing (not critical)")
        
        return True
    else:
        issues = []
        if missing_required:
            issues.append(f"{len(missing_required)} required packages missing")
        if not jupyter_ok:
            issues.append("Jupyter installation issues")
        if missing_dirs:
            issues.append(f"{len(missing_dirs)} module directories missing")
        
        print(f"❌ Setup issues detected: {', '.join(issues)}")
        
        # Provide instructions
        provide_setup_instructions(missing_required, missing_optional, 
                                 missing_dirs, missing_notebooks)
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)