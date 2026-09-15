import pandas as pd
import preprocessing.loader as ld
from preprocessing.dataset_config_info import DATASET_CONFIG
import preprocessing.inspector as insp
import preprocessing.analyzer as an
import preprocessing.cleaner as cl
"""
this is a pipeline pre for all the files for cic_ids2017
same logic as unsw_nb15
"""

#------------------------------------------------------------------
#first add al the files to a df (but add another column for source) 
#I dont wan to lose the data or make any assumptions
#------------------------------------------------------------------


def main() -> None:

    """load and analyze the UNSW-NB15
     dataset and the CIC-IDS2017 dataset,
      and print the reports for both datasets."""   
    
    dataset_name = "cic2017"

    config = DATASET_CONFIG[dataset_name]
    #get all files
    files=config["file_paths"]
    label_column = config["label_column"]
    #create df of dfs
    dfs=[]

    #now, trying to use the regular pipeline components 
    
    for file in files:

        df , report = an.analyze_dataset(
        file_path=file,
        dataset_name=dataset_name,
        label_column=config["label_column"],
        columns=config["columns"]
        )
        print("\nBefore cleaning:\n")
        an.print_report(report)

        #clean the df
        df_clean = cl.clean_dataset(df)
        
        #build new report
        report = insp.inspect_dataset(
            df=df_clean,
            label_column="Label" #cleaner strip spaces
        )

        print("\nAfter cleaning:\n")
        an.print_report(report)

        #add ref source file
        df_clean["source_file"] = file
        dfs.append(df_clean)

    #append to merged df
    df_merged = pd.concat(
     dfs,
     ignore_index=True
    )

    report = insp.inspect_dataset(
           df=df_merged,
           label_column="Label" #cleaner strip spaces
        )
        
    an.print_report(report)
    

        

        

       
         
  

    


    
   
  
    




if __name__ == "__main__":
    main()