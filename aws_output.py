"""AWS Lambda function that pulls predictions from RDS and 
constructs an HTML to display outputs in tables based on
location.  

Written by Gabe Seidl"""

import pymysql

endpoint = 'test-db.c74miacwgycz.us-west-1.rds.amazonaws.com'
username = 'admin'
password = 'admin_pw'
database_name = 'Predictions_DB'

connection = pymysql.connect(host = endpoint, user = username, passwd = password, db = database_name)

def lambda_handler(event, context):
    cursor = connection.cursor()
    
    connection.commit()
    cursor.execute('SELECT * from Predictions')
    brk = cursor.fetchall()
    cursor.execute('SELECT * from New_bedford')
    nbd = cursor.fetchall()
    cursor.execute('SELECT * from Salem')
    sal = cursor.fetchall()
    cursor.execute('SELECT * from La')
    las = cursor.fetchall()
    cursor.execute('SELECT * from Santa_barbara')
    sab = cursor.fetchall()
    
    names = ["Brooklyn, NY", "New Bedford, MA", "Salem, MA", "Los Angeles, CA", "Santa Barbara, CA"]
    
    rows = [brk, nbd, sal, las, sab]
    
    
    html = """
        <html>
        <body>
            <h1>15 Day Forecast</h1>
        """
    
    for i in range(5):
        html += """
            <h2>""" + names[i] + """</h2>
            <table>
                <tr style = 'outline: thin solid'>
                    <th>Date</th>
                    <th>Height</th>
                    <th>y-hat lower/y-hat upper</th>
                    <th>Period</th>
                    <th>y-hat lower/y-hat upper</th>
                <tr>"""

        for row in rows[i]:
            html += "<tr style = 'outline: thin solid'>" + "<td style = 'text-align:center'>" + ("{0}".format(row[0])) + "</td>" + "<td style = 'text-align:center'>" + ("{0}".format(row[1])) + "</td>" + "<td style = 'text-align:center'>" + ("{0}".format(row[2])) + "</td>" + "<td style = 'text-align:center'>" + ("{0}".format(row[3])) + "</td>"+ "<td style = 'text-align:center'>" + ("{0}".format(row[4])) + "</td>" + "</tr>"
    
        html += """</table><br></br>"""
    
    html += """ 
        <br></br>
        <img src = 'https://www.benchmarklabs.com/wp-content/uploads/2022/01/Benchmark-Labs-Logo-New-400PX-copy.png'>
        </body>
        </html>
        """
        
    
    return {
        'statusCode': 200,
        'body': html,
        'headers': {
            'Content-Type': 'text/html',
        }
    }
        
    
