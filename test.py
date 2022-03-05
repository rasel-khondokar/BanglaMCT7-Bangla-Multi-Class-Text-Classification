import re

a = "দেশের রাজনীতি দিনকে দিন পচে যাচ্ছে। पैरेनकाइमा कोशिकाएं . what a shame. সুস্থ থাকা দায়।"
a = "".join(i for i in a if i in [".","।"] or 2432 <= ord(i) <= 2559 or ord(i)== 32)
print(re.sub(' +', ' ', a))