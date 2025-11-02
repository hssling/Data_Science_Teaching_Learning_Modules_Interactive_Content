#!/usr/bin/env python3
"""
Module 2: Mathematics and Statistics Fundamentals - Quiz
Interactive quiz to test understanding of mathematical and statistical concepts
"""

import random
import time
import numpy as np

class StatisticsQuiz:
    """Interactive quiz for Module 2: Mathematics and Statistics Fundamentals"""

    def __init__(self):
        self.questions = {
            'multiple_choice': [
                {
                    'question': 'What is the primary purpose of the Central Limit Theorem?',
                    'options': [
                        'A) To calculate the exact mean of any distribution',
                        'B) To explain why the sampling distribution of means becomes normal',
                        'C) To determine if data is normally distributed',
                        'D) To calculate confidence intervals without sample data'
                    ],
                    'correct': 'B',
                    'explanation': 'The CLT states that the sampling distribution of sample means approaches normality as sample size increases, regardless of the population distribution.'
                },
                {
                    'question': 'In hypothesis testing, what does a Type I error represent?',
                    'options': [
                        'A) Failing to reject a false null hypothesis',
                        'B) Rejecting a true null hypothesis',
                        'C) Accepting a false alternative hypothesis',
                        'D) Correctly rejecting a false null hypothesis'
                    ],
                    'correct': 'B',
                    'explanation': 'Type I error (α) is rejecting H₀ when it is actually true - a false positive.'
                },
                {
                    'question': 'What is the relationship between variance and standard deviation?',
                    'options': [
                        'A) Variance is the square root of standard deviation',
                        'B) Standard deviation is the square root of variance',
                        'C) They are the same measure',
                        'D) Variance is twice the standard deviation'
                    ],
                    'correct': 'B',
                    'explanation': 'Standard deviation (σ) is the square root of variance (σ²), providing a measure in the same units as the original data.'
                },
                {
                    'question': 'Which of the following is a measure of central tendency?',
                    'options': [
                        'A) Variance',
                        'B) Standard deviation',
                        'C) Median',
                        'D) Range'
                    ],
                    'correct': 'C',
                    'explanation': 'Median is a measure of central tendency, while variance, standard deviation, and range are measures of dispersion/spread.'
                },
                {
                    'question': 'What does the p-value represent in hypothesis testing?',
                    'options': [
                        'A) The probability that the null hypothesis is true',
                        'B) The probability of observing the data (or more extreme) assuming H₀ is true',
                        'C) The probability that the alternative hypothesis is true',
                        'D) The significance level of the test'
                    ],
                    'correct': 'B',
                    'explanation': 'The p-value is the probability of observing data as extreme as (or more extreme than) what was observed, assuming the null hypothesis is true.'
                }
            ],
            'true_false': [
                {
                    'question': 'The mean is always more affected by outliers than the median.',
                    'correct': True,
                    'explanation': 'The mean is sensitive to extreme values, while the median is robust to outliers.'
                },
                {
                    'question': 'In a normal distribution, approximately 68% of data falls within one standard deviation of the mean.',
                    'correct': True,
                    'explanation': 'This is the empirical rule: 68% within 1σ, 95% within 2σ, 99.7% within 3σ of the mean.'
                },
                {
                    'question': 'Correlation implies causation.',
                    'correct': False,
                    'explanation': 'Correlation does not imply causation - there may be confounding variables or reverse causation.'
                },
                {
                    'question': 'A 95% confidence interval means there is a 95% probability that the true parameter falls within that interval.',
                    'correct': False,
                    'explanation': 'A 95% confidence interval means that if we repeated the sampling many times, 95% of such intervals would contain the true parameter.'
                }
            ],
            'calculation': [
                {
                    'question': 'Calculate the standard deviation for the dataset: [2, 4, 6, 8, 10]',
                    'correct_answer': 2.828,  # Approximately 2.828
                    'tolerance': 0.1,
                    'explanation': 'Mean = 6, variance = ((2-6)² + (4-6)² + (6-6)² + (8-6)² + (10-6)²)/5 = (16+4+0+4+16)/5 = 40/5 = 8, std = √8 ≈ 2.828'
                },
                {
                    'question': 'If a dataset has a mean of 50 and a standard deviation of 10, what percentage of data would you expect to fall between 30 and 70?',
                    'correct_answer': 99.7,
                    'tolerance': 0.1,
                    'explanation': '30 and 70 are 2 standard deviations below and above the mean (50±20). The empirical rule states that 99.7% of data falls within 3 standard deviations, so nearly all data (99.7%) falls within 2 standard deviations.'
                }
            ]
        }

        self.score = 0
        self.total_questions = 0
        self.start_time = None

    def display_welcome(self):
        """Display quiz welcome message"""
        print("=" * 70)
        print("📊 MODULE 2: MATHEMATICS AND STATISTICS FUNDAMENTALS - QUIZ")
        print("=" * 70)
        print()
        print("Test your understanding of mathematical and statistical concepts!")
        print("This quiz covers key concepts from Module 2.")
        print()
        print("📋 Quiz Structure:")
        print("• Multiple Choice Questions")
        print("• True/False Questions")
        print("• Calculation Problems")
        print()
        print("⏰ Time Limit: 40 minutes")
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

    def ask_calculation(self, question_data):
        """Ask a calculation problem"""
        print(f"\n{question_data['question']}")
        print("(Provide a numerical answer, round to 1 decimal place if needed)")

        while True:
            try:
                answer = float(input("Your answer: ").strip())
                break
            except ValueError:
                print("Please enter a valid number.")

        # Check if answer is within tolerance
        correct = question_data['correct_answer']
        tolerance = question_data['tolerance']
        is_correct = abs(answer - correct) <= tolerance

        if is_correct:
            print("✅ Correct!")
            self.score += 1
        else:
            print(f"❌ Incorrect. The correct answer is approximately {correct:.1f}")

        print(f"💡 {question_data['explanation']}")
        self.total_questions += 1

        input("\nPress Enter to continue...")

    def run_quiz(self):
        """Run the complete quiz"""
        self.display_welcome()
        self.start_time = time.time()

        # Shuffle questions for variety
        mc_questions = random.sample(self.questions['multiple_choice'], 3)
        tf_questions = random.sample(self.questions['true_false'], 2)
        calc_questions = random.sample(self.questions['calculation'], 1)

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
        print("CALCULATION PROBLEMS")
        print("="*50)

        for question in calc_questions:
            self.ask_calculation(question)

        # Calculate final score
        self.display_results()

    def display_results(self):
        """Display quiz results"""
        end_time = time.time()
        duration = end_time - self.start_time

        percentage = (self.score / self.total_questions) * 100

        print("\n" + "="*70)
        print("📊 QUIZ RESULTS")
        print("="*70)
        print()
        print(f"📊 Score: {self.score}/{self.total_questions} ({percentage:.1f}%)")
        print(f"⏱️  Time Taken: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print()

        # Performance assessment
        if percentage >= 90:
            print("🏆 Excellent! You have a strong grasp of statistical concepts.")
            print("   You're ready to apply these concepts to data analysis!")
        elif percentage >= 70:
            print("✅ Good job! You understand the core statistical principles well.")
            print("   Consider reviewing the calculation problems if you struggled there.")
        elif percentage >= 50:
            print("📚 Satisfactory. You have a basic understanding of statistics.")
            print("   We recommend reviewing Module 2 materials, especially probability distributions.")
        else:
            print("📖 Needs improvement. Please review Module 2 thoroughly.")
            print("   Focus on hypothesis testing, confidence intervals, and basic statistical measures.")

        print()
        print("📋 Key Topics to Review:")
        print("• Measures of central tendency and dispersion")
        print("• Probability distributions and the Central Limit Theorem")
        print("• Hypothesis testing and p-values")
        print("• Confidence intervals and their interpretation")
        print("• Correlation vs causation")
        print()
        print("🔄 Ready to retake the quiz? Run this script again!")
        print("="*70)

if __name__ == "__main__":
    quiz = StatisticsQuiz()
    quiz.run_quiz()
