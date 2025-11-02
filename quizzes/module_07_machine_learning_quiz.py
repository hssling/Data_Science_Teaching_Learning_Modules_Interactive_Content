#!/usr/bin/env python3
"""
Module 7: Machine Learning - Quiz
Interactive quiz to test understanding of machine learning concepts
"""

import random
import time

class MachineLearningQuiz:
    """Interactive quiz for Module 7: Machine Learning"""

    def __init__(self):
        self.questions = {
            'multiple_choice': [
                {
                    'question': 'What is the primary difference between supervised and unsupervised learning?',
                    'options': [
                        'A) Supervised learning requires more computational resources',
                        'B) Supervised learning uses labeled data, unsupervised learning uses unlabeled data',
                        'C) Unsupervised learning is always more accurate',
                        'D) Supervised learning only works with numerical data'
                    ],
                    'correct': 'B',
                    'explanation': 'Supervised learning uses labeled training data to learn a mapping function, while unsupervised learning finds hidden patterns in unlabeled data.'
                },
                {
                    'question': 'Which of the following is an example of a regression problem?',
                    'options': [
                        'A) Predicting customer churn (Yes/No)',
                        'B) Classifying emails as spam or not spam',
                        'C) Predicting house prices based on features',
                        'D) Grouping customers by purchasing behavior'
                    ],
                    'correct': 'C',
                    'explanation': 'Regression predicts continuous numerical values, while the other options are classification or clustering problems.'
                },
                {
                    'question': 'What does the "curse of dimensionality" refer to?',
                    'options': [
                        'A) Too many features making the model complex',
                        'B) Data becoming sparse in high-dimensional space',
                        'C) Models taking too long to train',
                        'D) Difficulty in visualizing high-dimensional data'
                    ],
                    'correct': 'B',
                    'explanation': 'In high-dimensional spaces, data points become increasingly sparse, making distance-based algorithms less effective.'
                },
                {
                    'question': 'Which metric is most appropriate for evaluating a highly imbalanced classification problem?',
                    'options': [
                        'A) Accuracy',
                        'B) Precision',
                        'C) F1-Score',
                        'D) Mean Absolute Error'
                    ],
                    'correct': 'C',
                    'explanation': 'F1-Score balances precision and recall, making it suitable for imbalanced datasets where accuracy can be misleading.'
                },
                {
                    'question': 'What is the main purpose of cross-validation?',
                    'options': [
                        'A) To speed up model training',
                        'B) To reduce model complexity',
                        'C) To assess model generalization performance',
                        'D) To handle missing data'
                    ],
                    'correct': 'C',
                    'explanation': 'Cross-validation helps assess how well a model generalizes to unseen data by testing on multiple data splits.'
                }
            ],
            'true_false': [
                {
                    'question': 'Overfitting occurs when a model performs well on training data but poorly on test data.',
                    'correct': True,
                    'explanation': 'Overfitting happens when a model learns noise in the training data instead of the underlying pattern.'
                },
                {
                    'question': 'Feature scaling is always necessary for tree-based algorithms like Random Forest.',
                    'correct': False,
                    'explanation': 'Tree-based algorithms are invariant to feature scaling since they only use threshold splits.'
                },
                {
                    'question': 'Gradient descent is guaranteed to find the global minimum for any loss function.',
                    'correct': False,
                    'explanation': 'Gradient descent can get stuck in local minima, especially for non-convex loss functions.'
                },
                {
                    'question': 'Ensemble methods like Random Forest are always better than individual models.',
                    'correct': False,
                    'explanation': 'While ensemble methods often perform better, they may not always outperform well-tuned individual models.'
                }
            ],
            'short_answer': [
                {
                    'question': 'Name three common techniques to prevent overfitting.',
                    'sample_answers': [
                        'cross-validation',
                        'regularization (l1/l2)',
                        'early stopping',
                        'dropout',
                        'data augmentation',
                        'simpler model architecture',
                        'pruning (for trees)'
                    ],
                    'explanation': 'Common overfitting prevention techniques include regularization, cross-validation, early stopping, dropout, and simpler models.'
                },
                {
                    'question': 'What are the key components of a confusion matrix?',
                    'sample_answers': [
                        'true positives (tp)',
                        'true negatives (tn)',
                        'false positives (fp)',
                        'false negatives (fn)'
                    ],
                    'explanation': 'A confusion matrix contains four key components: TP, TN, FP, and FN, which form the basis for most classification metrics.'
                }
            ]
        }

        self.score = 0
        self.total_questions = 0
        self.start_time = None

    def display_welcome(self):
        """Display quiz welcome message"""
        print("=" * 60)
        print("🤖 MODULE 7: MACHINE LEARNING - QUIZ")
        print("=" * 60)
        print()
        print("Test your understanding of machine learning concepts!")
        print("This quiz covers key algorithms and techniques from Module 7.")
        print()
        print("📋 Quiz Structure:")
        print("• Multiple Choice Questions")
        print("• True/False Questions")
        print("• Short Answer Questions")
        print()
        print("⏰ Time Limit: 45 minutes")
        print("📊 Passing Score: 70%")
        print()
        input("Press Enter to begin the quiz...")

    def ask_multiple_choice(self, question_data):
        """Ask a multiple choice question"""
        print(f"\n{question_data['question']}")
        print()

        for option in question_data['options']:
            print(f"  {option}")

        while True:
            answer = input("\nYour answer (A/B/C/D): ").strip().upper()
            if answer in ['A', 'B', 'C', 'D']:
                break
            print("Please enter A, B, C, or D.")

        is_correct = answer == question_data['correct']
        if is_correct:
            print("✅ Correct!")
            self.score += 1
        else:
            print(f"❌ Incorrect. The correct answer is {question_data['correct']}.")

        print(f"💡 {question_data['explanation']}")
        self.total_questions += 1

        input("\nPress Enter to continue...")

    def ask_true_false(self, question_data):
        """Ask a true/false question"""
        print(f"\nTrue or False: {question_data['question']}")

        while True:
            answer = input("Your answer (True/False): ").strip().lower()
            if answer in ['true', 'false', 't', 'f']:
                break
            print("Please enter True or False.")

        user_answer = answer.startswith('t')
        is_correct = user_answer == question_data['correct']

        if is_correct:
            print("✅ Correct!")
            self.score += 1
        else:
            correct_answer = "True" if question_data['correct'] else "False"
            print(f"❌ Incorrect. The correct answer is {correct_answer}.")

        print(f"💡 {question_data['explanation']}")
        self.total_questions += 1

        input("\nPress Enter to continue...")

    def ask_short_answer(self, question_data):
        """Ask a short answer question"""
        print(f"\n{question_data['question']}")
        print("(Provide 2-3 key points)")

        answer = input("Your answer: ").strip().lower()

        # Simple keyword matching
        correct_keywords = [kw.lower() for kw in question_data['sample_answers']]
        matched_keywords = sum(1 for kw in correct_keywords if kw in answer)

        # Award partial credit
        if matched_keywords >= 2:
            print("✅ Excellent answer!")
            self.score += 1
        elif matched_keywords == 1:
            print("✅ Good answer - you got some key points!")
            self.score += 0.5
        else:
            print("❌ Answer needs improvement.")

        print(f"💡 {question_data['explanation']}")
        print("Sample correct answers could include:")
        for sample in question_data['sample_answers'][:3]:
            print(f"  • {sample.title()}")

        self.total_questions += 1
        input("\nPress Enter to continue...")

    def run_quiz(self):
        """Run the complete quiz"""
        self.display_welcome()
        self.start_time = time.time()

        # Shuffle questions for variety
        mc_questions = random.sample(self.questions['multiple_choice'], 3)
        tf_questions = random.sample(self.questions['true_false'], 2)
        sa_questions = random.sample(self.questions['short_answer'], 1)

        # Ask questions
        print("\n" + "="*50)
        print("MULTIPLE CHOICE QUESTIONS")
        print("="*50)

        for question in mc_questions:
            self.ask_multiple_choice(question)

        print("\n" + "="*50)
        print("TRUE/FALSE QUESTIONS")
        print("="*50)

        for question in tf_questions:
            self.ask_true_false(question)

        print("\n" + "="*50)
        print("SHORT ANSWER QUESTIONS")
        print("="*50)

        for question in sa_questions:
            self.ask_short_answer(question)

        # Calculate final score
        self.display_results()

    def display_results(self):
        """Display quiz results"""
        end_time = time.time()
        duration = end_time - self.start_time

        percentage = (self.score / self.total_questions) * 100

        print("\n" + "="*60)
        print("🤖 QUIZ RESULTS")
        print("="*60)
        print()
        print(f"📊 Score: {self.score:.1f}/{self.total_questions} ({percentage:.1f}%)")
        print(f"⏱️  Time Taken: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print()

        # Performance assessment
        if percentage >= 90:
            print("🏆 Excellent! You have a strong grasp of machine learning concepts.")
            print("   You're ready to tackle advanced ML problems!")
        elif percentage >= 70:
            print("✅ Good job! You understand the core ML concepts well.")
            print("   Consider reviewing model evaluation and validation techniques.")
        elif percentage >= 50:
            print("📚 Satisfactory. You have a basic understanding of ML.")
            print("   We recommend reviewing Module 7 materials, especially algorithms.")
        else:
            print("📖 Needs improvement. Please review Module 7 thoroughly.")
            print("   Focus on supervised vs unsupervised learning and evaluation metrics.")

        print()
        print("📋 Key Topics to Review:")
        print("• Supervised vs unsupervised learning")
        print("• Model evaluation metrics and validation")
        print("• Overfitting and regularization techniques")
        print("• Ensemble methods and bias-variance tradeoff")
        print("• Cross-validation and hyperparameter tuning")
        print()
        print("🔄 Ready to retake the quiz? Run this script again!")
        print("="*60)

if __name__ == "__main__":
    quiz = MachineLearningQuiz()
    quiz.run_quiz()
