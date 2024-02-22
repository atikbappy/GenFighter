import numpy as np


class Result:
    successful_attacks = 0
    failed_attacks = 0
    skipped_attacks = 0
    queries = 0
    word_modification_rate = []

    def __init__(self, adversarials):
        self.total = len(adversarials)
        self.adversarials = adversarials
        self.compute_result()

    @property
    def total_attempted_attacks(self):
        return self.successful_attacks + self.failed_attacks

    @property
    def average_queries(self):
        return self.queries / self.total_attempted_attacks

    @property
    def average_word_modification_rate(self):
        return round(np.mean(self.word_modification_rate) * 100, 2)

    """
        Accuracy Under Attack / Post Attack Accuracy
    """
    @property 
    def paa(self): 
        post_attack_accuracy = (self.failed_attacks * 100) / self.total
        post_attack_accuracy = round(post_attack_accuracy, 2)
        return post_attack_accuracy

    """
        Attack Success Rate
    """
    @property 
    def asr(self): 
        attack_success_rate = (self.successful_attacks * 100) / self.total_attempted_attacks
        attack_success_rate = round(attack_success_rate, 2)
        return attack_success_rate

    def compute_result(self):
        for adv in self.adversarials:
            if np.argmax(adv[1]) == adv[5]: # adv[5] is the actual label
                self.queries += adv[4]["Victim Model Queries"]
                # Only successful attacks have "Word Modif. Rate"
                if "Word Modif. Rate" in adv[4]:
                    self.word_modification_rate.append(adv[4]["Word Modif. Rate"])
    
                if adv[2]:
                    self.successful_attacks += 1
                else:
                    self.failed_attacks += 1
            else:
                self.skipped_attacks += 1

    def print_stats(self):
        output = f"Total Attacked Instances: {self.total_attempted_attacks + self.skipped_attacks}\n"
        output += f"Successful Attacks: {self.successful_attacks}\n"
        output += f"Failed Attacks: {self.failed_attacks}\n"
        output += f"Skipped Attacks: {self.skipped_attacks}\n"
        output += f"Average queries: {self.average_queries}\n"
        output += f"Average word modif. rate: {self.average_word_modification_rate}\n"
        output += f"Accuracy Under Attack / Post Attack Accuracy: {self.paa}\n"
        output += f"Attack Success Rate: {self.asr}\n"
        print(output)
