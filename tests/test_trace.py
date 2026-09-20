import importlib.util
import unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/"scripts/audit_scores.py"
s=importlib.util.spec_from_file_location("trace_audit",p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class TraceMetricTests(unittest.TestCase):
 def test_ordered(self):
  r=m.verdict(3,0,[1,2]);self.assertTrue(r["correct"]);self.assertFalse(r["failure"])
 def test_target_only_misses_competitor(self):
  r=m.verdict(3,2,[1]);self.assertTrue(r["confused"]);self.assertFalse(r["ungrounded"])
 def test_full_view_must_enter_minimum(self):
  r=m.verdict(1,2,[3]);self.assertTrue(r["ungrounded"]);self.assertEqual(r["H"],-1)
 def test_tie_fails(self):self.assertTrue(m.verdict(2,1,[1])["failure"])
 def test_incorrect_excluded(self):self.assertFalse(m.verdict(0,1,[-1])["failure"])
 def test_missing_rival_rejected(self):
  with self.assertRaises(ValueError):m.verdict(1,0,[])
 def test_nonfinite_rejected(self):
  with self.assertRaises(ValueError):m.verdict(1,float("nan"),[0])
if __name__=="__main__":unittest.main()
