let allCustomers = [];


// ==================================================
// LOAD DASHBOARD
// ==================================================

async function loadDashboard() {

    const refreshButton =
        document.querySelector(".refresh-btn");

    try {

        // ------------------------------------------
        // Refresh button loading state
        // ------------------------------------------

        if (refreshButton) {
            refreshButton.disabled = true;
            refreshButton.textContent = "↻ Refreshing...";
        }


        // ------------------------------------------
        // Dashboard loading state
        // ------------------------------------------

        document.getElementById("risk").textContent = "—";
        document.getElementById("incremental").textContent = "—";
        document.getElementById("targeted").textContent = "—";
        document.getElementById("customers").textContent = "—";

        document.getElementById("personalizedCount").textContent = "—";
        document.getElementById("standardCount").textContent = "—";
        document.getElementById("noInterventionCount").textContent = "—";

        document.getElementById("targetingRate").textContent = "—";
        document.getElementById("averageCart").textContent = "—";
        document.getElementById("recoveryPerTarget").textContent = "—";


        // ------------------------------------------
        // Table loading state
        // ------------------------------------------

        document.getElementById("customerTable").innerHTML = `
            <tr>
                <td colspan="5" class="loading-row">
                    Loading recovery opportunities...
                </td>
            </tr>
        `;


        // ------------------------------------------
        // Fetch customers
        // ------------------------------------------

        const response = await fetch("/customers");

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const results = await response.json();

        allCustomers = results;

        updateDashboard(results);

    }


    // ==================================================
    // ERROR
    // ==================================================

    catch (error) {

        console.error(
            "RecoverAI loading error:",
            error
        );

        document.getElementById("risk").textContent = "Error";
        document.getElementById("incremental").textContent = "Error";
        document.getElementById("targeted").textContent = "Error";
        document.getElementById("customers").textContent = "Error";

        document.getElementById("personalizedCount").textContent = "—";
        document.getElementById("standardCount").textContent = "—";
        document.getElementById("noInterventionCount").textContent = "—";

        document.getElementById("targetingRate").textContent = "—";
        document.getElementById("averageCart").textContent = "—";
        document.getElementById("recoveryPerTarget").textContent = "—";

        document.getElementById("customerTable").innerHTML = `
            <tr>
                <td colspan="5" class="loading-row">
                    Unable to load recovery opportunities.
                </td>
            </tr>
        `;
    }


    // ==================================================
    // ALWAYS RUN AFTER SUCCESS OR ERROR
    // ==================================================

    finally {

        if (refreshButton) {
            refreshButton.disabled = false;
            refreshButton.textContent = "↻ Refresh";
        }
    }
}



// ==================================================
// UPDATE DASHBOARD
// ==================================================

function updateDashboard(results) {

    // ------------------------------------------
    // Total revenue at risk
    // ------------------------------------------

    const totalRisk = results.reduce(
        (sum, customer) =>
            sum + Number(customer.cart_amount),
        0
    );


    // ------------------------------------------
    // Targeted customers
    // ------------------------------------------

    const targetedCustomers = results.filter(
        customer =>
            customer.recommended_action !==
            "DO_NOT_INTERVENE"
    );


    // ------------------------------------------
    // Expected incremental revenue
    // ------------------------------------------

    const totalExpected = targetedCustomers.reduce(
        (sum, customer) =>
            sum + Number(customer.expected_value),
        0
    );


    const targeted =
        targetedCustomers.length;


    // ------------------------------------------
    // Strategy counts
    // ------------------------------------------

    const personalized = results.filter(
        customer =>
            customer.recommended_action ===
            "SEND_PERSONALIZED_REMINDER"
    ).length;


    const standard = results.filter(
        customer =>
            customer.recommended_action ===
            "SEND_STANDARD_REMINDER"
    ).length;


    const noIntervention = results.filter(
        customer =>
            customer.recommended_action ===
            "DO_NOT_INTERVENE"
    ).length;


    // ------------------------------------------
    // Main metrics
    // ------------------------------------------

    document.getElementById("risk").textContent =
        formatMoney(totalRisk);

    document.getElementById("incremental").textContent =
        formatMoney(totalExpected);

    document.getElementById("targeted").textContent =
        targeted;

    document.getElementById("customers").textContent =
        results.length;


    // ------------------------------------------
    // Strategy summary
    // ------------------------------------------

    document.getElementById("personalizedCount").textContent =
        personalized;

    document.getElementById("standardCount").textContent =
        standard;

    document.getElementById("noInterventionCount").textContent =
        noIntervention;


    // ------------------------------------------
    // Analytics
    // ------------------------------------------

    const targetingRate =
        results.length > 0
            ? (targeted / results.length) * 100
            : 0;


    const averageCart =
        results.length > 0
            ? totalRisk / results.length
            : 0;


    const recoveryPerTarget =
        targeted > 0
            ? totalExpected / targeted
            : 0;


    document.getElementById("targetingRate").textContent =
        targetingRate.toFixed(1) + "%";


    document.getElementById("averageCart").textContent =
        formatMoney(averageCart);


    document.getElementById("recoveryPerTarget").textContent =
        formatMoney(recoveryPerTarget);


    // ------------------------------------------
    // Customer table
    // ------------------------------------------

    renderTable(results);
}



// ==================================================
// RENDER TABLE
// ==================================================

function renderTable(results) {

    const table =
        document.getElementById("customerTable");

    table.innerHTML = "";


    const sortedCustomers = [...results]
        .sort(
            (a, b) =>
                Number(b.expected_value) -
                Number(a.expected_value)
        )
        .slice(0, 50);


    sortedCustomers.forEach(customer => {

        let actionClass = "none";


        if (
            customer.recommended_action ===
            "SEND_PERSONALIZED_REMINDER"
        ) {

            actionClass = "personalized";

        }


        else if (
            customer.recommended_action ===
            "SEND_STANDARD_REMINDER"
        ) {

            actionClass = "standard";

        }


        table.innerHTML += `
            <tr>

                <td>
                    ${customer.customer_id}
                </td>

                <td>
                    ${formatMoney(customer.cart_amount)}
                </td>

                <td>
                    ${
                        (
                            Number(
                                customer.recovery_probability
                            ) * 100
                        ).toFixed(0)
                    }%
                </td>

                <td>
                    ${formatMoney(customer.expected_value)}
                </td>

                <td class="action ${actionClass}">

                    <strong>
                        ${getActionLabel(
                            customer.recommended_action
                        )}
                    </strong>

                    <br>

                    <small>
                        ${getReason(customer)}
                    </small>

                </td>

            </tr>
        `;
    });
}



// ==================================================
// FILTERS
// ==================================================

function applyFilters() {

    const searchElement =
        document.getElementById("searchInput");


    const actionElement =
        document.getElementById("actionFilter");


    const search =
        searchElement
            ? searchElement.value.toLowerCase()
            : "";


    const action =
        actionElement
            ? actionElement.value
            : "ALL";


    const filtered =
        allCustomers.filter(customer => {

            const matchesSearch =
                String(customer.customer_id)
                    .toLowerCase()
                    .includes(search);


            const matchesAction =
                action === "ALL" ||
                customer.recommended_action === action;


            return (
                matchesSearch &&
                matchesAction
            );
        });


    renderTable(filtered);
}



// ==================================================
// ACTION LABEL
// ==================================================

function getActionLabel(action) {

    if (
        action ===
        "SEND_PERSONALIZED_REMINDER"
    ) {

        return "Personalized Reminder";

    }


    if (
        action ===
        "SEND_STANDARD_REMINDER"
    ) {

        return "Standard Reminder";

    }


    return "Do Not Intervene";
}



// ==================================================
// ACTION REASON
// ==================================================

function getReason(customer) {

    if (
        customer.recommended_action ===
        "SEND_PERSONALIZED_REMINDER"
    ) {

        return "High incremental revenue potential — prioritize customer.";

    }


    if (
        customer.recommended_action ===
        "SEND_STANDARD_REMINDER"
    ) {

        return "Positive incremental revenue potential — standard reminder.";

    }


    return "Low incremental value — avoid unnecessary intervention.";
}



// ==================================================
// MONEY FORMAT
// ==================================================

function formatMoney(value) {

    return (
        "₹" +
        Number(value).toLocaleString(
            "en-IN",
            {
                maximumFractionDigits: 2
            }
        )
    );
}



// ==================================================
// REFRESH
// ==================================================

function loadCustomers() {

    loadDashboard();

}



// ==================================================
// START DASHBOARD
// ==================================================

loadDashboard();