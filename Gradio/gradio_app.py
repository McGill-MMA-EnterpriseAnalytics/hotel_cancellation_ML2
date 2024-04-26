import gradio as gr
from gradio_calendar import Calendar
import pandas as pd
import datetime
from datetime import datetime, timedelta
import itertools
import pickle
import math
import numpy as np

import warnings
warnings.filterwarnings("ignore")

base_input = pd.read_csv('hotel_booking.csv')

with open('bestModel_hpo.pkl', 'rb') as f:
    data = pickle.load(f)
    loaded_model = data['model']

def get_predictions(arrival_date, checkout_date, country, meal, reserved_room_type, no_adults, no_children, no_babies, market_segment, distribution_channel):
    #arrival_date = '2024-04-30 00:00:00'
    #checkout_date = '2024-05-05 00:00:00'
    #arrival_date = datetime.strptime(arrival_date, '%Y-%m-%d %H:%M:%S')
    #checkout_date = datetime.strptime(checkout_date, '%Y-%m-%d %H:%M:%S')

    # Calculate lead time from today to arrival date
    today = datetime.today()
    lead_time = (arrival_date - today).days

    # Extract month name, week number, and day of month from arrival date
    arrival_date_month = arrival_date.strftime('%B')
    arrival_date_week_number = arrival_date.isocalendar()[1]
    arrival_date_day_of_month = arrival_date.day

    # Calculate the number of weekend nights and week nights
    weekend_days = {5, 6} 
    stays_in_weekend_nights = 0
    stays_in_week_nights = 0

    delta = timedelta(days=1)
    current_day = arrival_date
    while current_day < checkout_date:
        if current_day.weekday() in weekend_days:
            stays_in_weekend_nights += 1
        else:
            stays_in_week_nights += 1
        current_day += delta

    hotel = base_input['hotel'].unique().tolist()
    is_repeated_guest = base_input['is_repeated_guest'].unique().tolist()
    previous_cancellations = base_input['previous_cancellations'].unique().tolist()
    previous_bookings_not_canceled = math.ceil(base_input['previous_bookings_not_canceled'].mean())
    booking_changes = math.ceil(base_input['booking_changes'].mean())
    deposit_type = ['No Deposit', 'Non Refund', 'Refundable']
    days_in_waiting_list = math.ceil(base_input['days_in_waiting_list'].mean())
    customer_type = base_input['customer_type'].unique().tolist()
    adr = [x for x in range(50,200,50)]
    required_car_parking_spaces = math.ceil(base_input['required_car_parking_spaces'].mean())
    total_of_special_requests = math.ceil(base_input['total_of_special_requests'].mean())

    all_combinations = list(itertools.product(
    hotel,
    [lead_time],  
    [arrival_date_month],  
    [arrival_date_week_number],  
    [arrival_date_day_of_month],  
    [stays_in_weekend_nights],  
    [stays_in_week_nights],  
    [no_adults],
    [no_children],  
    [no_babies],  
    [meal],
    [country],
    [market_segment],
    [distribution_channel],
    is_repeated_guest,
    previous_cancellations,
    [previous_bookings_not_canceled],
    reserved_room_type,
    [booking_changes],
    deposit_type, 
    [days_in_waiting_list],  
    customer_type,
    adr,
    [required_car_parking_spaces],  
    [total_of_special_requests] 
))
    
    df = pd.DataFrame(all_combinations, columns=[
    'hotel',
    'lead_time',
    'arrival_date_month',
    'arrival_date_week_number',
    'arrival_date_day_of_month',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'babies',
    'meal',
    'country',
    'market_segment',
    'distribution_channel',
    'is_repeated_guest',
    'previous_cancellations',
    'previous_bookings_not_canceled',
    'reserved_room_type',
    'booking_changes',
    'deposit_type',
    'days_in_waiting_list',
    'customer_type',
    'adr',
    'required_car_parking_spaces',
    'total_of_special_requests'
])
    
    df['children'] = df['children'].astype(float)
    df['adr'] = df['adr'].astype(float)

    y_pred = loaded_model.predict_proba(df)
    cancellation_probability = y_pred[:, 1]
    df['cancellation_probability'] = cancellation_probability

    avg_cancellations = df['cancellation_probability'].mean()
    
    output_text = f"The average probability of cancellation for this booking is {round(100*avg_cancellations, 2)}%"

    return output_text, df

def get_prediction_with_breakdown(arrival_date, checkout_date, country, meal, reserved_room_type, no_adults, no_children, no_babies, market_segment, distribution_channel, breakdown_options):
    output_text, df = get_predictions(arrival_date, checkout_date, country, meal, reserved_room_type, no_adults, no_children, no_babies, market_segment, distribution_channel)
    breakdown_cols = [{'Deposit Type': 'deposit_type', 'Room Price': 'adr', 'Market Segment': 'market_segment', 'Hotel': 'hotel'}.get(item, item) for item in breakdown_options]
    if 'Room Price' in breakdown_options:
        df['adr'] = pd.cut(df['adr'], bins=[0, 50, 100, 200, 5400], labels=['Less than $50', 'Upto $100', 'Upto $200', 'Above $200'])

    output_table = df.groupby(breakdown_cols)['cancellation_probability'].mean().reset_index()
    output_table['cancellation_probability'] = output_table['cancellation_probability'].fillna(0) 
    output_table['cancellation_probability'] = np.round(output_table['cancellation_probability'] * 100,2)
    output_table = output_table.sort_values('cancellation_probability', ascending=False)
    output_table['cancellation_probability'] = output_table['cancellation_probability'].astype(str) + '%'
    output_table.columns = breakdown_options + ['Cancellation Probability']

    return output_table


def enable_output():
    return gr.Row(visible=True)


with gr.Blocks() as demo:
    gr.Markdown("## ANALYZE PROBABILITY OF CANCELLATIONS")
    gr.Markdown("<div style='text-align: center;'><p>Enter the Booking Details</p></div>")    
    with gr.Row():
        arrival_date = Calendar(type="datetime", label="Arrival Date")
        checkout_date = Calendar(type="datetime", label="Checkout Date")
        no_adults = gr.Number(label="Number of Adults", value=1)
        no_children = gr.Number(label="Number of Children", value=0)
        no_babies = gr.Number(label="Number of Babies", value=0)    
    with gr.Row():
        country = gr.Dropdown(label="Nationality", choices=base_input['country'].unique().tolist(), value='PRT')
        reserved_room_type = gr.Dropdown(label="Room Type", choices=base_input['reserved_room_type'].unique().tolist(), value='A')
        meal = gr.Dropdown(label="Meal Type", choices=[x for x in base_input['meal'].unique() if x != 'Undefined'], value='BB')
        market_segment = gr.Dropdown(label="Market Segment", choices=[x for x in base_input['market_segment'].unique() if x != 'Undefined'],value='Direct')
        distribution_channel = gr.Dropdown(label="Distribution Channel", choices=[x for x in base_input['distribution_channel'].unique() if x != 'Undefined'],value='Direct') 

    with gr.Row() as buttons:
        clear_button = gr.Button("Clear")
        submit_button = gr.Button("Submit")

    with gr.Row(visible=False) as output_row1:
        output_text1 = gr.Markdown()
        output_df = gr.DataFrame(visible=False)
        breakdown_options = gr.Dropdown(label="Select options for breakdown", choices=['Deposit Type','Room Price','Market Segment','Hotel'], value=['Deposit Type','Room Price'], multiselect=True)
        submit_button2 = gr.Button("Get Predictions")

    with gr.Row(visible=False) as output_row2:
        output_table = gr.DataFrame()
        
    
    submit_button.click(get_predictions,
                        inputs=[arrival_date, checkout_date, country, meal, reserved_room_type, no_adults, no_children, no_babies, market_segment, distribution_channel],
                        outputs=[output_text1, output_df])
    
    submit_button.click(enable_output, inputs=[], outputs=output_row1)

    submit_button2.click(get_prediction_with_breakdown,
                             inputs=[arrival_date, checkout_date, country, meal, reserved_room_type, no_adults, no_children, no_babies, market_segment, distribution_channel, breakdown_options],
                             outputs=output_table)
    
    submit_button2.click(enable_output, inputs=[], outputs=output_row2)

    clear_button.click(None, js="window.location.reload()")
    
demo.launch(share=True)
