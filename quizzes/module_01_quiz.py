#!/usr/bin/env python3
"""
Module 1: Introduction to Data Science - Quiz
Interactive quiz to test understanding of data science fundamentals
"""

import random
import time

class DataScienceQuiz:
    """Interactive quiz for Module 1: Introduction to Data Science"""

    def __init__(self):
        self.questions = {
            'multiple_choice': [
                {
                    'question': 'What is the primary goal of data science?',
                    'options': [
                        'A) To collect as much data as possible',
                        'B) To extract meaningful insights and knowledge from data',
                        'C) To create complex algorithms',
                        'D) To build data storage systems'
                    ],
                    'correct': 'B',
                    'explanation': 'Data science focuses on extracting actionable insights from data to solve real-world problems.'
                },
                {
                    'question': 'Which of the following is NOT a key component of the data science process?',
                    'options': [
                        'A) Data collection',
                        'B) Data cleaning',
                        'C) Algorithm development',
                        'D) Hardware maintenance'
                    ],
                    'correct': 'D',
                    'explanation': 'Hardware maintenance is not part of the core data science workflow.'
                },
                {
                    'question': 'What does ETL stand for in data processing?',
                    'options': [
                        'A) Extract, Transform, Load',
                        'B) Evaluate, Test, Learn',
                        'C) Explore, Train, Launch',
                        'D) Examine, Transform, Label'
                    ],
                    'correct': 'A',
                    'explanation': 'ETL stands for Extract, Transform, Load - the process of moving data from source to destination.'
                },
                {
                    'question': 'Which type of data is characterized by structured format and easily searchable?',
                    'options': [
                        'A) Structured data',
                        'B) Semi-structured data',
                        'C) Unstructured data',
                        'D) Raw data'
                    ],
                    'correct': 'A',
                    'explanation': 'Structured data is organized in a predefined format, typically in databases with rows and columns.'
                },
                {
                    'question': 'What is the main difference between supervised and unsupervised learning?',
                    'options': [
                        'A) Supervised learning uses labeled data, unsupervised learning uses unlabeled data',
                        'B) Supervised learning is faster than unsupervised learning',
                        'C) Unsupervised learning requires more computational resources',
                        'D) Supervised learning only works with numerical data'
                    ],
                    'correct': 'A',
                    'explanation': 'Supervised learning uses labeled training data, while unsupervised learning finds patterns in unlabeled data.'
                }
            ],
            'true_false': [
                {
                    'question': 'Data science and machine learning are the same thing.',
                    'correct': False,
                    'explanation': 'Data science is broader and includes machine learning as one of its tools.'
                },
                {
                    'question': 'Big data refers only to the volume of data.',
                    'correct': False,
                    'explanation': 'Big data is characterized by the 5 V\'s: Volume, Velocity, Variety, Veracity, and Value.'
                },
                {
                    'question': 'Python is the most popular programming language for data science.',
                    'correct': True,
                    'explanation': 'Python has become the dominant language for data science due to its rich ecosystem of libraries.'
                },
                {
                    'question': 'Data visualization is only used at the end of a data science project.',
                    'correct': False,
                    'explanation': 'Data visualization is used throughout the data science process for exploration and communication.'
                }
            ],
            'short_answer': [
                {
                    'question': 'Name three key skills required for a data scientist.',
                    'sample_answers': [
                        'programming (python/r)',
                        'statistics and mathematics',
                        'domain knowledge',
                        'communication skills',
                        'machine learning',
                        'data visualization'
                    ],
                    'explanation': 'Key skills include technical skills (programming, stats, ML) and soft skills (communication, domain knowledge).'
                },
                {
                    'question': 'What are the main steps in the CRISP-DM methodology?',
                    'sample_answers': [
                        'business understanding',
                        'data understanding',
                        'data preparation',
                        'modeling',
                        'evaluation',
                        'deployment'
                    ],
                    'explanation': 'CRISP-DM includes: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment.'
                }
            ]
        }

        self.score = 0
        self.total_questions = 0
        self.start_time = None

    def display_welcome(self):
        """Display quiz welcome message"""
        print("=" * 60)
        print("🎯 MODULE 1: INTRODUCTION TO DATA SCIENCE - QUIZ")
        print("=" * 60)
        print()
        print("Test your understanding of data science fundamentals!")
        print("This quiz covers key concepts from Module 1.")
        print()
        print("📋 Quiz Structure:")
        print("• Multiple Choice Questions")
        print("• True/False Questions")
        print("• Short Answer Questions")
        print()
        print("⏰ Time Limit: 30 minutes")
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

        # Simple keyword matching (in a real implementation, this would be more sophisticated)
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
        print("🎯 QUIZ RESULTS")
        print("="*60)
        print()
        print(f"📊 Score: {self.score:.1f}/{self.total_questions} ({percentage:.1f}%)")
        print(f"⏱️  Time Taken: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print()

        # Performance assessment
        if percentage >= 90:
            print("🏆 Excellent! You have a strong understanding of data science fundamentals.")
            print("   You're ready to move on to more advanced topics!")
        elif percentage >= 70:
            print("✅ Good job! You understand the core concepts well.")
            print("   Consider reviewing areas where you struggled.")
        elif percentage >= 50:
            print("📚 Satisfactory. You have a basic understanding.")
            print("   We recommend reviewing Module 1 materials.")
        else:
            print("📖 Needs improvement. Please review Module 1 thoroughly.")
            print("   Focus on the fundamental concepts before proceeding.")

        print()
        print("📋 Key Topics to Review:")
        print("• Data science definition and scope")
        print("• Data science process and methodologies")
        print("• Types of data and their characteristics")
        print("• Machine learning fundamentals")
        print("• Required skills and tools")
        print()
        print("🔄 Ready to retake the quiz? Run this script again!")
        print("="*60)

if __name__ == "__main__":
    quiz = DataScienceQuiz()
    quiz.run_quiz()
