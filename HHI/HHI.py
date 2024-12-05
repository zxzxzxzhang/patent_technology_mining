import pandas as pd
df = pd.read_excel('data.xlsx')
df.astype(str)

yearly_patent_count = df.groupby('Application Year')['Publication Number'].count()
yearly_patent_count = pd.DataFrame(yearly_patent_count).rename(columns={'Publication Number': 'Patent Count'})

# Count the number of patents per IPC subclass and year
ipc_yearly_patent_count = df[df['IPC Main Classification'] != '-'].groupby(
    ['Application Year', 'IPC Main Classification']
)['Publication Number'].count().unstack(fill_value=0)

ipc_yearly_patent_count.rename(columns={'IPC Main Classification': 'Application Year'}, inplace=True)
ipc_yearly_patent_count = ipc_yearly_patent_count.reset_index()
yearly_patent_count = yearly_patent_count.reset_index()


full_ipc_yearly_patent_count = ipc_yearly_patent_count.merge(yearly_patent_count, on='Application Year', how='left')


def hhiyear(ipc, count):
    if count == 0:
        return 0
    hhi = 0
    for share in ipc:
        hhi += (share / count) ** 2
    return hhi

# Calculate HHI for each year
hhi_results = []
for index, row in full_ipc_yearly_patent_count.iterrows():
    ipc = row.iloc[1:-1]
    count = row['Patent Count']
    hhi = hhiyear(ipc, count)
    hhi_results.append(hhi)

# Add HHI and clean up the dataset
full_ipc_yearly_patent_count['HHI'] = hhi_results
full_ipc_yearly_patent_count['Application Year'] = full_ipc_yearly_patent_count['Application Year'].astype(int)
full_ipc_yearly_patent_count.rename(columns={'Application Year': 'Year'}, inplace=True)

full_ipc_yearly_patent_count.to_excel('HHI.xlsx', index=False)
