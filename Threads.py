import threading, queue

from Helper import *

workingQueue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

def QueueThreadHandler():
  global stopQueueThread, queueThread, workingQueue

  while (True):
    if (not workingQueue.empty()):
      item = workingQueue.get()
      if (item is not None):
        item = list(item)
        task = item.pop(0)
        print("Working with:", task)
        print("Configurations:", item)
        if (task == "CLS_INF"):  # Classification Inference.
          MakeFolders()
          thread = threading.Thread(target=ManipulateClassificationInferenceThread, args=(item,))
        elif (task == "CLS_PRS"):  # Classification Processing.
          MakeFolders()
          thread = threading.Thread(target=ManipulateClassificationThread, args=(item,))
        elif (task == "WSI_CNV"):  # WSI Conversion.
          MakeFolders()
          thread = threading.Thread(target=ManipulateWSIConversion, args=(item,))
        elif (task == "SEG_WEK_PRS"):  # Weakly Supervised Segmentation using U-Net.
          MakeFolders()
          thread = threading.Thread(target=ManipulateWeaklyUNetSegmentation, args=(item,))
        elif (task == "SEG_WEK_INF"):  # Weakly Supervised Segmentation Inference using U-Net.
          MakeFolders()
          thread = threading.Thread(target=ManipulateWeaklyUNetSegmentationInferenceThread, args=(item,))
        else:
          continue
        thread.start()  # Start thread.
        thread.join()  # Wait for thread to finish.
        workingQueue.task_done()  # Remove item from queue.
    else:
      print("Queue is empty.")
      stopQueueThread = True
      break
