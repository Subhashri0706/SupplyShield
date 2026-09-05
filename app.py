
from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from google import genai

app = Flask(__name__)

# Load data
suppliers = pd.read_csv("data/suppliers.csv")
inventory = pd.read_csv("data/inventory.csv")
shipments = pd.read_csv("data/shipments.csv")
orders = pd.read_csv("data/orders.csv")

# Gemini client
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)


def understand_notice(notice):

    prompt = f"""
Read this supply chain disruption notice.

Identify:
supplier_id
product_id
shipment_id
disruption_type
delay_days

Return ONLY valid JSON.

Example:
{{
    "supplier_id": "S001",
    "product_id": "P001",
    "shipment_id": "SH001",
    "disruption_type": "Production Delay",
    "delay_days": 7
}}

Notice:
{notice}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    text = response.text.strip()

    text = text.replace("```json", "")
    text = text.replace("```", "")

    return json.loads(text)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    notice = request.form.get("notice", "")

    try:

        # --------------------------------
        # STEP 1: Understand notice
        # --------------------------------

        extracted = understand_notice(notice)

        shipment_id = extracted.get("shipment_id")
        product_id = extracted.get("product_id")
        delay_days = int(extracted.get("delay_days", 0))

        # --------------------------------
        # STEP 2: Find shipment
        # --------------------------------

        shipment = shipments[
            shipments["shipment_id"] == shipment_id
        ]

        if shipment.empty:
            return jsonify({
                "message": "No matching shipment found."
            })

        shipment_row = shipment.iloc[0]

        supplier_id = shipment_row["supplier_id"]

        # --------------------------------
        # STEP 3: Find supplier
        # --------------------------------

        supplier = suppliers[
            suppliers["supplier_id"] == supplier_id
        ]

        # --------------------------------
        # STEP 4: Find inventory
        # --------------------------------

        product_inventory = inventory[
            inventory["product_id"] == product_id
        ]

        total_stock = int(
            product_inventory["stock_units"].sum()
        )

        # --------------------------------
        # STEP 5: Find affected orders
        # --------------------------------

        product_orders = orders[
            orders["product_id"] == product_id
        ].copy()

        total_ordered = int(
            product_orders["quantity"].sum()
        )

        remaining_stock = max(
            total_stock - total_ordered,
            0
        )

        # --------------------------------
        # STEP 6: Calculate shortage
        # --------------------------------

        shortage = max(
            total_ordered - total_stock,
            0
        )

        # --------------------------------
        # STEP 7: Calculate affected orders
        # --------------------------------

        affected_orders = []

        available = total_stock

        # High priority first
        priority_order = {
            "High": 1,
            "Medium": 2,
            "Low": 3
        }

        product_orders["priority_rank"] = (
            product_orders["priority"]
            .map(priority_order)
        )

        product_orders = product_orders.sort_values(
            by=["priority_rank", "required_date"]
        )

        for _, order in product_orders.iterrows():

            quantity = int(order["quantity"])

            if available >= quantity:

                status = "Covered"

                available -= quantity

            else:

                status = "At Risk"

                affected_orders.append({
                    "order_id": order["order_id"],
                    "customer": order["customer"],
                    "product_id": order["product_id"],
                    "quantity": quantity,
                    "required_date": order["required_date"],
                    "priority": order["priority"],
                    "status": status
                })

        # --------------------------------
        # STEP 8: If shipment delay creates risk
        # --------------------------------

        if delay_days > 0 and len(affected_orders) == 0:

            # Check orders whose required date
            # comes before delayed shipment arrival

            expected_date = pd.to_datetime(
                shipment_row["expected_date"]
            )

            delayed_date = expected_date + timedelta(
                days=delay_days
            )

            for _, order in product_orders.iterrows():

                required_date = pd.to_datetime(
                    order["required_date"]
                )

                if required_date < delayed_date:

                    if not any(
                        x["order_id"] == order["order_id"]
                        for x in affected_orders
                    ):

                        affected_orders.append({
                            "order_id": order["order_id"],
                            "customer": order["customer"],
                            "product_id": order["product_id"],
                            "quantity": int(order["quantity"]),
                            "required_date": order["required_date"],
                            "priority": order["priority"],
                            "status": "At Risk"
                        })

        # --------------------------------
        # STEP 9: Impact level
        # --------------------------------

        if shortage > 0 or len(affected_orders) >= 3:

            impact_level = "High"

        elif len(affected_orders) > 0:

            impact_level = "Medium"

        else:

            impact_level = "Low"

        # --------------------------------
        # STEP 10: Dates
        # --------------------------------

        expected_date = pd.to_datetime(
            shipment_row["expected_date"]
        )

        delayed_date = expected_date + timedelta(
            days=delay_days
        )

        expected_date_text = expected_date.strftime(
            "%Y-%m-%d"
        )

        delayed_date_text = delayed_date.strftime(
            "%Y-%m-%d"
        )

        # --------------------------------
        # STEP 11: Recommendation
        # --------------------------------

        if shortage > 0:

            recommendation = (
                "Prioritize high-priority customer orders, "
                "consider expedited replenishment, and "
                "inform customers whose orders cannot be "
                "fulfilled on time."
            )

        elif len(affected_orders) > 0:

            recommendation = (
                "Prioritize the affected high-priority orders "
                "and consider reallocating available stock. "
                "Customers facing delays should be informed."
            )

        else:

            recommendation = (
                "Current inventory can cover existing customer "
                "orders. Monitor the delayed shipment closely "
                "and prepare a contingency plan if the delay "
                "increases."
            )

        # --------------------------------
        # STEP 12: Final result
        # --------------------------------

        result = {

            "extracted_information": extracted,

            "supplier":
                supplier.to_dict("records"),

            "shipment":
                shipment.to_dict("records"),

            "inventory":
                product_inventory.to_dict("records"),

            "affected_orders":
                affected_orders,

            "impact": {

                "impact_level":
                    impact_level,

                "orders_at_risk":
                    len(affected_orders),

                "number_of_affected_orders":
                    len(affected_orders),

                "total_stock":
                    total_stock,

                "total_ordered":
                    total_ordered,

                "remaining_stock":
                    remaining_stock,

                "total_shortage":
                    shortage,

                "delayed_shipment_quantity":
                    int(shipment_row["quantity"]),

                "delay_days":
                    delay_days,

                "expected_arrival":
                    expected_date_text,

                "delayed_arrival":
                    delayed_date_text

            },

            "recommendation":
                recommendation
        }

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )

