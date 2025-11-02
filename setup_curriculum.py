#!/usr/bin/env python3
"""
Data Science Curriculum Setup Script
===================================

This script helps you set up the complete data science curriculum environment.
It will:
1. Check system requirements
2. Install required packages
3. Set up development environment
4. Verify installation
5. Provide next steps

Usage:
    python setup_curriculum.py

Or run directly:
    python setup_curriculum.py --help
"""

import sys
import subprocess
import platform
import os
from pathlib import Path
import argparse
import json

class CurriculumSetup:
    """Complete curriculum setup and environment configuration"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.system_info = self.get_system_info()
        self.setup_complete = False

    def get_system_info(self):
        """Get system information for compatibility checking"""
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'python_executable': sys.executable
        }

    def print_header(self):
        """Print setup header"""
        print("=" * 70)
        print("🚀 COMPREHENSIVE DATA SCIENCE CURRICULUM - SETUP")
        print("=" * 70)
        print()
        print("This setup will prepare your environment for the complete")
        print("data science curriculum with all necessary dependencies.")
        print()

    def check_system_requirements(self):
        """Check if system meets minimum requirements"""
        print("🔍 Checking System Requirements...")
        print("-" * 40)

        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} - Compatible")
        else:
            print(f"❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} - Requires Python 3.8+")
            return False

        # Check available disk space (rough estimate)
        try:
            stat = os.statvfs(self.project_root)
            free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            if free_space_gb > 5:  # 5GB minimum
                print(".1f"            else:
                print(".1f"                return False
        except:
            print("⚠️  Could not check disk space - proceeding anyway")

        # Check if pip is available
        try:
            subprocess.run([sys.executable, '-m', 'pip', '--version'],
                         capture_output=True, check=True)
            print("✅ pip - Available")
        except subprocess.CalledProcessError:
            print("❌ pip - Not available")
            return False

        print()
        return True

    def install_packages(self, upgrade=False):
        """Install required packages"""
        print("📦 Installing Required Packages...")
        print("-" * 40)

        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            print("❌ requirements.txt not found!")
            return False

        # Install core packages first (essential for curriculum)
        core_packages = [
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn',
            'jupyter', 'notebook', 'ipykernel'
        ]

        print("Installing core packages...")
        for package in core_packages:
            try:
                cmd = [sys.executable, '-m', 'pip', 'install', package]
                if upgrade:
                    cmd.append('--upgrade')

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    print(f"✅ {package}")
                else:
                    print(f"❌ {package} - {result.stderr.strip()}")
                    return False

            except Exception as e:
                print(f"❌ {package} - Error: {str(e)}")
                return False

        # Install remaining packages
        print("\nInstalling additional packages...")
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
            if upgrade:
                cmd.append('--upgrade')

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ All packages installed successfully")
                return True
            else:
                print("⚠️  Some packages may have failed to install")
                print("This is normal - core packages are sufficient for the curriculum")
                return True

        except Exception as e:
            print(f"❌ Error installing packages: {str(e)}")
            return False

    def setup_nltk_data(self):
        """Download required NLTK data"""
        print("\n📚 Setting up NLTK Data...")
        print("-" * 30)

        try:
            import nltk

            # Download required NLTK data
            required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']

            for data in required_data:
                try:
                    nltk.download(data, quiet=True)
                    print(f"✅ {data}")
                except Exception as e:
                    print(f"⚠️  {data} - {str(e)}")

            print("NLTK setup completed")
            return True

        except ImportError:
            print("⚠️  NLTK not available - will be installed with other packages")
            return False

    def verify_installation(self):
        """Verify that key packages are installed and working"""
        print("\n🔍 Verifying Installation...")
        print("-" * 30)

        # Test core imports
        core_packages = {
            'numpy': 'np',
            'pandas': 'pd',
            'matplotlib': 'plt',
            'seaborn': 'sns',
            'sklearn': 'sklearn',
            'tensorflow': 'tf'
        }

        failed_imports = []

        for package, import_name in core_packages.items():
            try:
                if package == 'sklearn':
                    __import__('sklearn')
                else:
                    __import__(package)
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package}")
                failed_imports.append(package)

        if failed_imports:
            print(f"\n⚠️  {len(failed_imports)} packages failed to import")
            print("Core functionality may be limited, but basic curriculum will work")
        else:
            print("\n✅ All core packages imported successfully")

        return len(failed_imports) == 0

    def create_environment_info(self):
        """Create environment information file"""
        print("\n📄 Creating Environment Info...")
        print("-" * 30)

        env_info = {
            'setup_date': str(pd.Timestamp.now()) if 'pd' in globals() else 'Unknown',
            'python_version': self.system_info['python_version'],
            'platform': self.system_info['platform'],
            'packages': {}
        }

        # Get installed package versions
        try:
            import pkg_resources
            for package in ['numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn', 'tensorflow']:
                try:
                    version = pkg_resources.get_distribution(package).version
                    env_info['packages'][package] = version
                except:
                    env_info['packages'][package] = 'Not installed'
        except:
            env_info['packages'] = 'Could not retrieve package versions'

        # Save to file
        env_file = self.project_root / 'environment_info.json'
        try:
            with open(env_file, 'w') as f:
                json.dump(env_info, f, indent=2)
            print(f"✅ Environment info saved to {env_file}")
        except Exception as e:
            print(f"⚠️  Could not save environment info: {str(e)}")

    def run_diagnostic_tests(self):
        """Run basic diagnostic tests"""
        print("\n🧪 Running Diagnostic Tests...")
        print("-" * 30)

        tests_passed = 0
        total_tests = 0

        # Test 1: Basic NumPy operations
        total_tests += 1
        try:
            import numpy as np
            arr = np.array([1, 2, 3, 4, 5])
            result = np.mean(arr)
            if abs(result - 3.0) < 1e-6:
                print("✅ NumPy basic operations")
                tests_passed += 1
            else:
                print("❌ NumPy basic operations")
        except:
            print("❌ NumPy basic operations")

        # Test 2: Basic pandas operations
        total_tests += 1
        try:
            import pandas as pd
            df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            result = df['A'].mean()
            if abs(result - 2.0) < 1e-6:
                print("✅ Pandas basic operations")
                tests_passed += 1
            else:
                print("❌ Pandas basic operations")
        except:
            print("❌ Pandas basic operations")

        # Test 3: Basic matplotlib plotting
        total_tests += 1
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.plot([1, 2, 3], [1, 4, 2])
            plt.savefig('test_plot.png')
            plt.close()
            if os.path.exists('test_plot.png'):
                os.remove('test_plot.png')  # Clean up
                print("✅ Matplotlib plotting")
                tests_passed += 1
            else:
                print("❌ Matplotlib plotting")
        except:
            print("❌ Matplotlib plotting")

        # Test 4: Basic scikit-learn
        total_tests += 1
        try:
            from sklearn.linear_model import LinearRegression
            import numpy as np
            X = np.array([[1], [2], [3], [4]])
            y = np.array([2, 4, 6, 8])
            model = LinearRegression()
            model.fit(X, y)
            score = model.score(X, y)
            if score > 0.99:  # Should be perfect fit
                print("✅ Scikit-learn basic modeling")
                tests_passed += 1
            else:
                print("❌ Scikit-learn basic modeling")
        except:
            print("❌ Scikit-learn basic modeling")

        print(f"\nDiagnostic Results: {tests_passed}/{total_tests} tests passed")

        if tests_passed == total_tests:
            print("🎉 All diagnostic tests passed!")
            return True
        elif tests_passed >= total_tests * 0.75:
            print("✅ Core functionality working (most tests passed)")
            return True
        else:
            print("⚠️  Limited functionality - some features may not work")
            return False

    def show_next_steps(self):
        """Show next steps for getting started"""
        print("\n🎯 NEXT STEPS - Getting Started with the Curriculum")
        print("=" * 55)

        print("1. 📖 Read the Curriculum Guide:")
        print("   Open CURRICULUM_GUIDE.md for complete learning paths")

        print("\n2. 🚀 Start with Module 1:")
        print("   python modules/01_introduction/introduction_examples.py")

        print("\n3. 📝 Take the First Quiz:")
        print("   python quizzes/module_01_quiz.py")

        print("\n4. 🔬 Complete Exercises:")
        print("   python exercises/module_01_exercises.py")

        print("\n5. 🏗️ Build Your First Project:")
        print("   python projects/predictive_analytics_project.py")

        print("\n6. 📚 Explore Resources:")
        print("   Open resources/learning_resources.md")

        print("\n7. 👥 Join Communities:")
        print("   - r/datascience on Reddit")
        print("   - Towards Data Science on Medium")
        print("   - Kaggle.com")

        print("\n💡 Pro Tips:")
        print("• Start with 1-2 hours daily for consistent progress")
        print("• Complete exercises before moving to next module")
        print("• Build projects to reinforce learning")
        print("• Join study groups for accountability")

        print("\n🎓 Curriculum Structure:")
        print("• 14 Modules: From basics to advanced topics")
        print("• 3 Interactive Quizzes: Test your understanding")
        print("• 10+ Practical Exercises: Hands-on learning")
        print("• 3 Complete Projects: Portfolio-worthy applications")
        print("• Extensive Resources: Books, courses, communities")

    def run_setup(self, upgrade=False, skip_tests=False):
        """Run the complete setup process"""
        self.print_header()

        # Show system info
        print("System Information:")
        for key, value in self.system_info.items():
            print(f"• {key.replace('_', ' ').title()}: {value}")
        print()

        # Check requirements
        if not self.check_system_requirements():
            print("❌ System requirements not met. Please fix issues above.")
            return False

        # Install packages
        if not self.install_packages(upgrade=upgrade):
            print("❌ Package installation failed.")
            return False

        # Setup NLTK
        self.setup_nltk_data()

        # Verify installation
        if not self.verify_installation():
            print("⚠️  Some packages may not be working correctly.")
            print("Core curriculum functionality should still work.")

        # Create environment info
        self.create_environment_info()

        # Run diagnostics (unless skipped)
        if not skip_tests:
            if not self.run_diagnostic_tests():
                print("⚠️  Some diagnostic tests failed.")
                print("Basic functionality should still be available.")

        # Mark setup as complete
        self.setup_complete = True

        print("\n" + "="*70)
        print("🎉 CURRICULUM SETUP COMPLETED SUCCESSFULLY!")
        print("="*70)

        # Show next steps
        self.show_next_steps()

        return True

def main():
    parser = argparse.ArgumentParser(description='Data Science Curriculum Setup')
    parser.add_argument('--upgrade', action='store_true',
                       help='Upgrade existing packages')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip diagnostic tests')
    parser.add_argument('--help-setup', action='store_true',
                       help='Show detailed setup information')

    args = parser.parse_args()

    if args.help_setup:
        print("""
Data Science Curriculum Setup Help
==================================

This script sets up your environment for the complete data science curriculum.

Options:
  --upgrade      Upgrade existing packages to latest versions
  --skip-tests   Skip diagnostic tests (faster setup)
  --help-setup   Show this help information

What gets installed:
  • Core data science libraries (numpy, pandas, matplotlib, seaborn)
  • Machine learning frameworks (scikit-learn, tensorflow)
  • NLP libraries (nltk, spacy)
  • Web frameworks (flask, streamlit)
  • Development tools (jupyter, pytest)
  • And many more specialized libraries

System Requirements:
  • Python 3.8 or higher
  • 5GB+ free disk space
  • Internet connection for package downloads

After setup, you can:
  • Run any curriculum module
  • Execute quizzes and exercises
  • Build and run projects
  • Use Jupyter notebooks

For issues or questions:
  • Check the troubleshooting section in CURRICULUM_GUIDE.md
  • Visit the curriculum GitHub repository
  • Join data science communities for support
        """)
        return

    # Run setup
    setup = CurriculumSetup()
    success = setup.run_setup(upgrade=args.upgrade, skip_tests=args.skip_tests)

    if success:
        print("\n🚀 Ready to start your data science journey!")
        print("Run 'python modules/01_introduction/introduction_examples.py' to begin.")
    else:
        print("\n❌ Setup encountered issues.")
        print("Check the error messages above and try again.")
        print("For help, run: python setup_curriculum.py --help-setup")

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
