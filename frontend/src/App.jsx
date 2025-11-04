import { useState } from "react";

function App() {
  const [member, setMember] = useState("[A superposition of all STG members]");

  const selectNewMember = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/randommember");
      const data = await response.json();
      setMember(data.member);  // Update the state with the random member
    } catch (error) {
      console.error("Error fetching member:", error);
    }
  };

  return (
    <div>
      <h1>UBC Solar x Hack the Track</h1>
      <p>Click the button to choose a random STG member!</p>
      <button onClick={selectNewMember}>Get Random Member</button>
      <p>Chosen member: {member}</p>
    </div>
  );
}

export default App;
