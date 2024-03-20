import React, { useState } from 'react'
import axios from "axios"
import doc1 from '../assets/doc1.jpg'

const Predict = () => {
    const [prediction, setPrediction] = useState("")
    const [answer, setAnswer] = useState("")

    const handleChange = (event) => {
        setPrediction(event.target.value);
    };

    console.log(prediction)
    const handleClick= async (e) => {
        console.log(prediction);
        try {
            const response = await axios.post('http://127.0.0.1:5000/process',{
                input: prediction,
            },{
                headers: {
                  "Content-Type": "application/json",
                  "Access-Control-Allow-Origin": "*",
                },
              })
            console.log(response.data)
            setAnswer(response.data.final_prediction)
        }
        
         catch (error) {
            console.log(error)
        }
    }
  return (
    <>
        <div className='inline-flex align-middle items-center'>
        {/* <div className="flex lg:w-1/4 lg:h-1/4 w-full mb-10 lg:mb-0 rounded-lg overflow-hidden align-middle items-center">
            <img
              alt="feature"
              className="object-cover object-center h-full w-full"
              src={doc1}
            />
          </div> */}
          <div>
            <div className='align-center text-center lg:mt-12 mt:auto font-sans'>
                <h3 className='font-bold text-2xl font-sans'>Enter your Diseases as a string</h3>
                <input type='text' name='symptoms' id='symptoms' value={prediction} className='my-6 px-2 py-2 border-2 border-black hover:shadow-xl lg:mt-4 mt-auto' onChange={handleChange} placeholder='Enter Symptoms'/>
            </div>
            <button type='submit' onClick={handleClick} className='font-medium text-xl border border-black bg-[#FF7F50] mx-4 mb-4 py-2 px-4 rounded-lg' placeholder='Helllllloooo'>Submit Your input</button>
            { answer ? (
                <div className=' font-sans'>
                The prediction according to your inputs is that you have: <h1 className='font-semibold text-[#FF0000] underline'>{answer}</h1>
            </div>
            ) : ( <div>
                Nothing Here
                </div>
            )

            }
          </div>
        </div>
    </>
  )
}

export default Predict
