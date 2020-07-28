# -*- coding: UTF-8 -*-
import sys
import Libs.faultage_detect as faultage_detect

if __name__ == '__main__':
  result = faultage_detect.analysis_faultage(sys.argv)
  if result != '':
    print('Error:', result)
  else:
    print('Execute success!')