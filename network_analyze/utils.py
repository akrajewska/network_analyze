import datetime as dt


def dateparse(date):
    #'%d/%m/%Y %H:%M'
    '2017-07-07 03:30:00'
    try:
        return dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

    except:
        return dt.datetime.strptime(date, '%d/%m/%Y %H:%M')
