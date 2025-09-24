from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import pickle
import pandas as pd
import os

app = FastAPI()


kapha_model = pickle.load(open("kapha_model.pkl", "rb"))
pitta_model = pickle.load(open("pitta_model.pkl", "rb"))
vata_model = pickle.load(open("vata_model.pkl", "rb"))


level_map = {0: "Normal", 1: "Medium", 2: "High"}


templates = Jinja2Templates(directory="templates")



@app.get("/", response_class=HTMLResponse)
def show_form(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "form_submitted": False, "show_form": True}
    )



@app.post("/", response_class=HTMLResponse)
def predict(
    request: Request,
    age: int = Form(...),
    gender: str = Form(...),
    sleepquality: int = Form(...),
    stresslevel: int = Form(...),
    appetite: int = Form(...),
    energylevel: int = Form(...),
    angerlevel: int = Form(...),
    forgetfulness: int = Form(...),
    anxietylevel: int = Form(...),
    boweltype: str = Form(...),
    cravings: str = Form(...),
    skintype: str = Form(...),
    hairtype: str = Form(...),
    bodyframe: str = Form(...),
    sweating: str = Form(...),
):
    print(f"Received form data: age={age}, gender={gender}, sleepquality={sleepquality}")
    print(f"stresslevel={stresslevel}, appetite={appetite}, energylevel={energylevel}")
    print(f"angerlevel={angerlevel}, forgetfulness={forgetfulness}, anxietylevel={anxietylevel}")
    print(f"boweltype={boweltype}, cravings={cravings}, skintype={skintype}")
    print(f"hairtype={hairtype}, bodyframe={bodyframe}, sweating={sweating}")

    form_data = {
        "Age": age,
        "Gender": gender,
        "SleepQuality": sleepquality,
        "StressLevel": stresslevel,
        "Appetite": appetite,
        "EnergyLevel": energylevel,
        "AngerLevel": angerlevel,
        "Forgetfulness": forgetfulness,
        "AnxietyLevel": anxietylevel,
        "BowelType": boweltype,
        "Cravings": cravings,
        "SkinType": skintype,
        "HairType": hairtype,
        "BodyFrame": bodyframe,
        "Sweating": sweating,
    }

    input_df = pd.DataFrame([form_data])

    
    kapha_pred_raw = kapha_model.predict(input_df)[0]
    pitta_pred_raw = pitta_model.predict(input_df)[0]
    vata_pred_raw = vata_model.predict(input_df)[0]

    kapha_pred = level_map.get(kapha_pred_raw, f"Unknown ({kapha_pred_raw})")
    pitta_pred = level_map.get(pitta_pred_raw, f"Unknown ({pitta_pred_raw})")
    vata_pred = level_map.get(vata_pred_raw, f"Unknown ({vata_pred_raw})")

    
    # Store results in a simple way (in a real app, you'd use sessions or database)
    global last_results
    last_results = {
        "kapha": kapha_pred,
        "pitta": pitta_pred,
        "vata": vata_pred
    }
    
    return RedirectResponse(url="/results", status_code=303)


@app.get("/results", response_class=HTMLResponse)
def show_results(request: Request):
    global last_results
    if 'last_results' not in globals():
        return RedirectResponse(url="/")
    
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "kapha": last_results["kapha"],
            "pitta": last_results["pitta"],
            "vata": last_results["vata"],
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Use 0.0.0.0 for production (when PORT is set by Render), 127.0.0.1 for local development
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    uvicorn.run(app, host=host, port=port)
